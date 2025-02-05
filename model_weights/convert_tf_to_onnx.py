import os
import tf2onnx
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model, Sequential
from tensorflow_addons import layers as La


def unfold(inp):
    """
    Splits a three-digit integer into separate values for kernel sizes or strides.
    
    Args:
        inp (int): A three-digit integer.

    Returns:
        tuple: Three separate integers.
    """
    return inp // 100, (inp % 100) // 10, inp % 10


class ConvBlock(L.Layer):
    """
    A convolutional block containing two Conv3D layers, instance normalization, and ReLU activation.
    Supports optional upsampling via Conv3DTranspose.
    """
    def __init__(self, opt, filters, kernel, strides=111,
                 up=False, up_kernel=None, up_strides=None, **kwargs):
        """
        Initializes the ConvBlock.
        
        Args:
            opt: Configuration options containing weight decay.
            filters (int): Number of filters in the convolutional layers.
            kernel (int): Kernel size as a three-digit integer.
            strides (int): Stride size as a three-digit integer.
            up (bool, optional): If True, enables upsampling. Defaults to False.
            up_kernel (int, optional): Kernel size for upsampling. Defaults to None.
            up_strides (int, optional): Stride size for upsampling. Defaults to None.
        """
        super(ConvBlock, self).__init__(**kwargs)
        kernel = unfold(kernel)
        strides = unfold(strides)

        regu = l2(opt.weight_decay)
        self.conv1 = L.Conv3D(
            filters, kernel, strides, "same", use_bias=False, kernel_regularizer=regu,
            name="conv1")
        self.norm1 = La.InstanceNormalization(name="norm1")
        self.conv2 = L.Conv3D(
            filters, kernel, (1, 1, 1), "same", use_bias=False, kernel_regularizer=regu,
            name="conv2")
        self.norm2 = La.InstanceNormalization(name="norm2")
        self.relu = L.ReLU()

        if up:
            up_kernel = unfold(up_kernel)
            up_strides = unfold(up_strides)

            self.cat = L.Concatenate()
            self.up = L.Conv3DTranspose(
                filters, up_kernel, up_strides, "same", use_bias=False, kernel_regularizer=regu,
                name="up")

    def call(self, x, training=True, **kwargs):
        """
        Forward pass of the convolutional block.
        
        Args:
            x: Input tensor.
            training (bool, optional): Whether in training mode. Defaults to True.
        
        Returns:
            Tensor: Processed output.
        """
        if hasattr(self, "up"):
            y, x = x
            x = self.cat([y, self.up(x)])
        if isinstance(x, tuple):
            x, guide = x
            x = self.relu(self.norm1(self.conv1(x)) + guide)
            x = self.relu(self.norm2(self.conv2(x)))
        else:
            x = self.relu(self.norm1(self.conv1(x)))
            x = self.relu(self.norm2(self.conv2(x)))
        return x


class DINs(Model):
    """
    Deep Interactive Networks (DINs) model for medical image segmentation.
    """
    def __init__(self, opt, logger, name="model"):
        """
        Initializes the DINs model.
        
        Args:
            opt: Configuration options.
            logger: Logger instance.
            name (str, optional): Model name. Defaults to "model".
        """
        super(DINs, self).__init__(name=name)
        self.init_channel = opt.init_channel
        self.max_channels = opt.max_channels

        def get_channel(layer):
            return min(self.init_channel * 2 ** layer, self.max_channels)

        self.down1 = ConvBlock(opt, get_channel(0), 133, 111, name=f"{name}/down1")
        self.down2 = ConvBlock(opt, get_channel(1), 133, 122, name=f"{name}/down2")
        self.down3 = ConvBlock(opt, get_channel(2), 333, 122, name=f"{name}/down3")
        self.down4 = ConvBlock(opt, get_channel(3), 333, 122, name=f"{name}/down4")
        self.bridge = ConvBlock(opt, get_channel(4), 333, 222, name=f"{name}/bridge")
        self.up4 = ConvBlock(opt, get_channel(3), 333, 111, True, 222, 222, name=f"{name}/up4")
        self.up3 = ConvBlock(opt, get_channel(2), 333, 111, True, 122, 122, name=f"{name}/up3")
        self.up2 = ConvBlock(opt, get_channel(1), 133, 111, True, 122, 122, name=f"{name}/up2")
        self.up1 = ConvBlock(opt, get_channel(0), 133, 111, True, 122, 122, name=f"{name}/up1")

        regu = l2(opt.weight_decay)
        self.dim = Sequential(layers=[
            L.MaxPool3D((1, 8, 8), (1, 8, 8)),
            L.Conv3D(get_channel(4), (3, 3, 3), (2, 2, 2), "same", use_bias=True, kernel_regularizer=regu)
        ], name="dim")

        self.final = L.Conv3D(
            opt.n_class, (1, 1, 1), use_bias=True, kernel_regularizer=regu, name=f"{name}/final")


    def call(self, x, training=True, **kwargs):
        """
        Forward pass of the DINs model.
        
        Args:
            x: Tuple containing image and guide tensors.
            training (bool, optional): Whether in training mode. Defaults to True.
        
        Returns:
            Tensor: Model output.
        """
        image, guide = x
        x = tf.concat((image, guide), axis=-1)
        small_guide = self.dim(guide)

        x1 = self.down1(x, training)
        x2 = self.down2(x1, training)
        x3 = self.down3(x2, training)
        x4 = self.down4(x3, training)
        x = self.bridge((x4, small_guide), training)
        x = self.up4((x4, x), training)
        x = self.up3((x3, x), training)
        x = self.up2((x2, x), training)
        x = self.up1((x1, x), training)
        x = self.final(x)

        return x

    def get_config(self):
        return {
            "init_channel": self.init_channel,
            "max_channels": self.max_channels
        }


# Define the root directory containing the folds
class Opt:
    """
    A simple configuration class that initializes attributes from a dictionary.

    Args:
        options (Dict[str, Any]): Dictionary containing key-value pairs of configuration parameters.
    """
    def __init__(self, options):
        for key, value in options.items():
            setattr(self, key, value)


if __name__ == "__main__":
    # Define the root directory containing the trained model folds
    root_dir = "DINs_finetuned"
    input_shape = [
        (None, 10, 512, 160, 1), # Shape for image input
        (None, 10, 512, 160, 2) # Shape for guidance input
    ]
    
    # Define model configuration parameters
    opt_dict = {
        "init_channel": 30,
        "max_channels": 320,
        "n_class": 2,
        "weight_decay": 3e-5,
        "gamma": 1.,
        
    }

    opt = Opt(opt_dict)

    # Iterate through each fold directory
    for fold in os.listdir(root_dir):
        fold_path = os.path.join(root_dir, fold)
        ckpt_dir = os.path.join(fold_path, "bestckpt")

        if os.path.isdir(ckpt_dir):
            print(f"Processing: {fold}")

            # Create a dummy model instance with the same structure as the original
            model = DINs(opt=opt, logger=None)  # Provide correct opt and logger if needed
            model.build(input_shape)
            
            # Find the latest checkpoint in the directory
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
            
            # Load model weights from the checkpoint
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(ckpt_path).expect_partial()
            print(f"Successfully restored model from {ckpt_path}")
            
            # Convert the loaded model to ONNX format
            input_signature = [
                tf.TensorSpec(shape=input_shape[0], dtype=tf.float32, name="image"),
                tf.TensorSpec(shape=input_shape[1], dtype=tf.float32, name="guide")
            ]

            @tf.function(input_signature=[input_signature])
            def model_inference(input_data):
                return model(input_data, training=False)

            onnx_model_path = os.path.join(fold_path, "checkpoint.onnx")

            model_proto, external_tensor_storage = tf2onnx.convert.from_function(
                model_inference,
                input_signature=[input_signature],
                opset=13,  # Use an appropriate ONNX opset version
                output_path=onnx_model_path
            )

            print(f"Successfully converted {fold} to {onnx_model_path}")

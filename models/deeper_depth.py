from models.base_model import *
from models.nets import *
from models.backbone import *


class DepthEstimation(BaseModel):
    def __init__(self):
        super().__init__()

    def build_model(self, input_shape, output_shape):
        super().build_model(input_shape, output_shape)
        input_tensor = Input(input_shape + (3,))
        feature_map = res50(input_tensor, activation=relu)
        x = conv_layer(feature_map, 1024, (1, 1), activation=relu)
        x = fast_up_projection(x, 512, relu)
        x = fast_up_projection(x, 256, relu)
        x = fast_up_projection(x, 128, relu)
        x = fast_up_projection(x, 64, relu)
        x = conv_layer(x, 3, (3, 3), activation=relu, batch_norm=False)

        model = Model(input_tensor, x)

        self.model = model
        self.model.summary(160)

        return


depth = DepthEstimation()
depth.build_model((224, 224), (112, 112, 3))

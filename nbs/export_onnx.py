from deimkit.exporter import Exporter
from deimkit.config import Config

config = Config("outputs/plantdoc/deim_hgnetv2_m_30ep_416x416/config.yml")
exporter = Exporter(config)

output_path = exporter.to_onnx(
    checkpoint_path="outputs/plantdoc/deim_hgnetv2_m_30ep_416x416/best.pth",
    output_path="outputs/plantdoc/deim_hgnetv2_m_30ep_416x416/model.onnx",
)
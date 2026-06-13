__author__ = "Johnny Godoy"
__license__ = "MIT"
__email__ = "johnny.godoy@ing.uchile.cl"
__maintainer__ = "Johnny Godoy"
__status__ = "Development"
__version__ = "0.1.0"

from compression_knn.knn import CompressionKNNClassifier
from compression_knn.knn import CompressionKNNClassifierCV
from compression_knn.preprocessor import VectorToTextTransformer

__all__ = [
	"CompressionKNNClassifier",
	"CompressionKNNClassifierCV",
	"VectorToTextTransformer",
]

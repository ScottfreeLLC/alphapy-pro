__version__ = "3.1.1"

# Suppress common sklearn warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
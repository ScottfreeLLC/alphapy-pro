__version__ = "3.1.1"

# Suppress common warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='No further splits with positive gain')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
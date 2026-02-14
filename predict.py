import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt  

#  Define the EXACT same model architecture 

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.network(x)


#  Load trained model with error handling

def load_model(model_path='mnist_model_state_dict.pth'):
    """Load trained MNIST model with proper error handling"""
    device = torch.device("cpu")
    model = MNISTClassifier().to(device)
    
    try:
        # Load the saved state_dict
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f" Model loaded successfully from '{model_path}'!")
        return model, device
    except FileNotFoundError:
        print(f" Error: Model file '{model_path}' not found!")
        print("   Please train the model first using train.py")
        return None, None
    except Exception as e:
        print(f" Error loading model: {e}")
        return None, None


#  Preprocessing pipeline (more accurate)

def preprocess_image(image_path):
    """
    Preprocess image to match MNIST format exactly:
    - 28x28 pixels
    - Grayscale
    - White digit on black background (MNIST format)
    - Pixel values normalized to [0, 1]
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28 (MNIST standard size)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # MNIST has white digit (255) on black background (0)
       
        background_value = img_array[0, 0]  # Top-left pixel
        if background_value > 127:  # Light background
            img_array = 255 - img_array  # Invert colors
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Add batch and channel dimensions: [1, 1, 28, 28]
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor, img_array
    
    except FileNotFoundError:
        print(f" Error: Image file '{image_path}' not found!")
        return None, None
    except Exception as e:
        print(f" Error preprocessing image: {e}")
        return None, None


#  Prediction function with visualization

def predict_digit(model, device, image_path, show_plot=True):
    """Predict digit from image with visualization"""
    
    # Preprocess image
    img_tensor, img_array = preprocess_image(image_path)
    if img_tensor is None:
        return None, None
    
    
    img_tensor = img_tensor.to(device)
    
    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get top-3 predictions
        top_probs, top_labels = torch.topk(probabilities, 3)
        
        predicted_label = top_labels[0][0].item()
        confidence = top_probs[0][0].item()
    
    # Visualization
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Show preprocessed image
        ax1.imshow(img_array, cmap='gray')
        ax1.set_title(f'Preprocessed Image\nPrediction: {predicted_label} ({confidence:.1%})')
        ax1.axis('off')
        
        # Show probability distribution
        probs = probabilities[0].cpu().numpy()
        digits = range(10)
        colors = ['green' if i == predicted_label else 'gray' for i in digits]
        ax2.bar(digits, probs, color=colors)
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Probability')
        ax2.set_title('Prediction Probabilities')
        ax2.set_xticks(digits)
        
        plt.tight_layout()
        plt.show()
    
    return predicted_label, confidence


#  Batch prediction for multiple images

def predict_batch(model, device, image_paths):
    """Predict digits for multiple images"""
    results = []
    
    for image_path in image_paths:
        digit, confidence = predict_digit(model, device, image_path, show_plot=False)
        results.append({
            'image': image_path,
            'prediction': digit,
            'confidence': confidence
        })
    
    return results


#  Interactive mode

def interactive_predict():
    """Interactive prediction mode"""
    model, device = load_model()
    if model is None:
        return
    
    print("\n" + "="*50)
    print(" MNIST DIGIT PREDICTOR")
    print("="*50)
    
    while True:
        print("\n Enter image path (or 'quit' to exit): ", end='')
        image_path = input().strip()
        
        if image_path.lower() in ['quit', 'q', 'exit']:
            print(" Goodbye!")
            break
        
        digit, confidence = predict_digit(model, device, image_path)
        if digit is not None:
            print(f" Prediction: {digit} (Confidence: {confidence:.2%})")


#  Test with MNIST test set 

def test_with_mnist_sample():
    """Test the model on a real MNIST test image"""
    try:
        import torchvision
        import torchvision.transforms as transforms
        
        # Load MNIST test set
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, transform=transform, download=True
        )
        
       
        idx = np.random.randint(len(test_dataset))
        img_tensor, true_label = test_dataset[idx]
        
        
        img_tensor = img_tensor.unsqueeze(0)
        
        # Predict
        model, device = load_model()
        if model is None:
            return
        
        with torch.no_grad():
            output = model(img_tensor.to(device))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_label = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_label].item()
        
        # Display
        plt.figure(figsize=(6, 6))
        plt.imshow(img_tensor.squeeze(), cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label} ({confidence:.1%})')
        plt.axis('off')
        plt.show()
        
    except ImportError:
        print("  torchvision not installed. Skipping MNIST test.")


#  Main execution

if __name__ == "__main__":
  
    model, device = load_model()
    
    if model is not None:
        
        print("\n Testing with random MNIST test image...")
        test_with_mnist_sample()
        
      
        image_path = "my_digit.png"  
        print(f"\n Predicting '{image_path}'...")
        digit, conf = predict_digit(model, device, image_path)
        if digit is not None:
            print(f" Final Prediction: {digit} (Confidence: {conf:.2%})")
        

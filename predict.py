import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from model import create_model

class UricAcidPredictor:
    """Predictor class for uric acid concentration from test images"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = create_model(num_classes=1, dropout_rate=0.5)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded on {self.device}")
    
    def predict_single_image(self, image_path):
        """Predict uric acid concentration for a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            predicted_ppm = prediction.squeeze().cpu().item()
        
        return predicted_ppm, image
    
    def get_random_test_samples(self, test_path, samples_per_label=1):
        """Get random samples from each label folder in test directory"""
        samples = {}
        
        # Get all label folders
        label_folders = [f for f in os.listdir(test_path) 
                        if os.path.isdir(os.path.join(test_path, f))]
        
        # Sort labels numerically
        label_folders.sort(key=lambda x: float(x))
        
        for label_folder in label_folders:
            label_path = os.path.join(test_path, label_folder)
            
            # Get all image files in this label folder
            image_files = [f for f in os.listdir(label_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(image_files) >= samples_per_label:
                # Randomly select samples
                selected_files = random.sample(image_files, samples_per_label)
                samples[float(label_folder)] = []
                
                for img_file in selected_files:
                    img_path = os.path.join(label_path, img_file)
                    samples[float(label_folder)].append(img_path)
        
        return samples
    
    def create_prediction_visualization(self, test_path, output_path='prediction_results.png'):
        """Create YOLO-style visualization with predictions"""
        
        # Get random samples from each label
        samples = self.get_random_test_samples(test_path, samples_per_label=1)
        
        if not samples:
            print("No test samples found!")
            return
        
        # Sort by label for consistent ordering
        sorted_labels = sorted(samples.keys())
        
        # Create subplot grid (adjust based on number of labels)
        n_labels = len(sorted_labels)
        if n_labels <= 6:
            rows, cols = 2, 3
        elif n_labels <= 9:
            rows, cols = 3, 3
        elif n_labels <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        fig.suptitle('Uric Acid Concentration Predictions', fontsize=16, fontweight='bold')

        plt.subplots_adjust(top=0.85)
        
        # Flatten axes array for easier indexing
        if rows == 1:
            axes = [axes]
        elif rows > 1 and cols > 1:
            axes = axes.flatten()
        
        prediction_results = []
        
        for idx, true_label in enumerate(sorted_labels):
            if idx >= len(axes):
                break
                
            # Get image path
            img_path = samples[true_label][0]
            
            # Make prediction
            predicted_ppm, original_image = self.predict_single_image(img_path)
            
            # Store results
            prediction_results.append({
                'true_label': true_label,
                'predicted_ppm': predicted_ppm,
                'error': abs(predicted_ppm - true_label),
                'image_path': img_path
            })
            
            # Display image with prediction
            ax = axes[idx] if isinstance(axes, (list, np.ndarray)) else axes
            ax.imshow(original_image)
            ax.axis('off')
            
            # Create prediction text with color coding
            error = abs(predicted_ppm - true_label)
            if error <= 5:
                color = 'green'
                status = '✓'
            elif error <= 10:
                color = 'orange' 
                status = '⚠'
            else:
                color = 'red'
                status = '✗'
            
            # Title with prediction info
            title = f'{status} True: {true_label:.0f} ppm\nPred: {predicted_ppm:.1f} ppm\nError: {error:.1f} ppm'
            ax.set_title(title, fontsize=15, color=color, fontweight='bold', pad=20)
        
        # Hide unused subplots
        for idx in range(len(sorted_labels), len(axes)):
            if isinstance(axes, (list, np.ndarray)):
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        
        errors = [result['error'] for result in prediction_results]
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        print(f"Mean Absolute Error: {mean_error:.2f} ± {std_error:.2f} ppm")
        print(f"Error Range: {min_error:.2f} - {max_error:.2f} ppm")
        
        # Accuracy by thresholds
        accurate_5ppm = sum(1 for e in errors if e <= 5) / len(errors) * 100
        accurate_10ppm = sum(1 for e in errors if e <= 10) / len(errors) * 100
        
        print(f"Accuracy (±5 ppm): {accurate_5ppm:.1f}%")
        print(f"Accuracy (±10 ppm): {accurate_10ppm:.1f}%")
        
        print("\nDETAILED RESULTS:")
        print("-" * 60)
        for result in prediction_results:
            status = "✓" if result['error'] <= 5 else "⚠" if result['error'] <= 10 else "✗"
            print(f"{status} Label: {result['true_label']:3.0f} ppm | "
                  f"Predicted: {result['predicted_ppm']:6.1f} ppm | "
                  f"Error: {result['error']:5.1f} ppm")
        
        return prediction_results

def main():
    """Main function to run predictions"""
    # Configuration
    model_path = 'best_uric_acid_model.pth'
    test_path = 'data/0724_Enlarged/test'
    output_path = 'uric_acid_predictions.png'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please ensure you have trained the model first.")
        return
    
    # Check if test directory exists
    if not os.path.exists(test_path):
        print(f"Error: Test directory '{test_path}' not found!")
        return
    
    # Create predictor and run predictions
    try:
        predictor = UricAcidPredictor(model_path)
        results = predictor.create_prediction_visualization(test_path, output_path)
        
        print(f"\nVisualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Add src folder to path to import our modules
sys.path.append('src')

# ============================================================================
# 🎯 CONFIGURATION - CHANGE THESE VALUES TO TEST DIFFERENT IMAGES
# ============================================================================

# IMAGE TO TEST - Just change this line to test different images!
IMAGE_NAME = "g1.jpg"  # Options: "g1.jpg", "mm1.jpg", "nt1.jpg", "p1.jpg"

# OR use full path if image is elsewhere
# IMAGE_PATH = "C:/path/to/your/image.jpg"  # Uncomment and modify if needed

# Model and class names paths (usually don't need to change these)
MODEL_PATH = "best_brain_tumor_model_enhanced.keras"
CLASS_NAMES_PATH = "class_names.json"

# ============================================================================
# 🧠 BRAIN TUMOR CLASSIFICATION PIPELINE VISUALIZER
# ============================================================================

class ReportVisualizer:
    def __init__(self, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        """Initialize visualizer with trained model and class names."""
        print(f"🔄 Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        print(f"🔄 Loading class names from: {class_names_path}")
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        print(f"✅ Model and class names loaded: {self.class_names}")
    
    def load_image(self, image_path):
        """Load an image for testing."""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        print(f"📁 Loading image: {os.path.basename(image_path)}")
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        # Ensure image is float32 and in [0, 255] range initially for display consistency
        image = tf.cast(image, tf.float32) 
        # For initial display, we want the raw image. Normalization to [0,1] happens later.
        # So, if image was uint8, casting to float32 keeps values in [0, 255] which imshow handles.
        return image
    
    def get_preprocessing_steps(self, image):
        """Apply preprocessing step by step and capture each stage."""
        pipeline_steps = []
        
        # Step 1: Original
        original = tf.identity(image)
        pipeline_steps.append({'image': original, 'title': '1. Original Image'})

        # Step 2: Resize
        resized = tf.image.resize(image, [128, 128])
        pipeline_steps.append({'image': resized, 'title': '2. Resized (128x128)'})
        
        # Step 3: Normalize (THIS IS WHERE IT GOES TO [0,1])
        normalized = tf.cast(resized, tf.float32) / 255.0
        pipeline_steps.append({'image': normalized, 'title': '3. Normalized [0,1]'})
        
        # Step 4: Enhance Contrast
        contrast_enhanced = tf.image.adjust_contrast(normalized, contrast_factor=1.5)
        pipeline_steps.append({'image': contrast_enhanced, 'title': '4. Contrast Enhanced'})
        
        # Final image for the model
        final_image = tf.image.adjust_brightness(contrast_enhanced, delta=0.0)
        
        return pipeline_steps, final_image

    def get_augmentation_examples(self, processed_image):
        """Create examples of each augmentation type."""
        augmentation_examples = []
        flip_layer = tf.keras.layers.RandomFlip("horizontal")
        rotation_layer = tf.keras.layers.RandomRotation(0.1)
        translation_layer = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
        batch_image = tf.expand_dims(processed_image, axis=0)

        augmentation_examples.append({'image': processed_image, 'title': 'Original (No Aug.)'})
        augmentation_examples.append({'image': tf.squeeze(flip_layer(batch_image, training=True)), 'title': 'Random Flip'})
        augmentation_examples.append({'image': tf.squeeze(rotation_layer(batch_image, training=True)), 'title': 'Random Rotation'})
        augmentation_examples.append({'image': tf.squeeze(translation_layer(batch_image, training=True)), 'title': 'Random Translation'})
        
        return augmentation_examples
    
    def make_prediction(self, processed_image):
        """Make prediction on a processed image."""
        input_batch = tf.expand_dims(processed_image, axis=0)
        predictions = self.model.predict(input_batch, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_idx]
        confidence = predictions[0][predicted_idx] * 100
        all_probs = {self.class_names[i]: predictions[0][i] * 100 for i in range(len(self.class_names))}
        return predicted_class, confidence, all_probs

    def save_visual(self, fig, base_name, image_name_tag):
        """Saves the figure with a report-friendly name."""
        save_path = f"{base_name}_{image_name_tag}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig) # Close the figure to free memory
        print(f"💾 Figure saved as: {save_path}")

    def generate_visuals(self, image_path):
        """Orchestrates the creation and saving of all report visuals."""
        image = self.load_image(image_path)
        if image is None: return

        base_image_name = os.path.splitext(os.path.basename(image_path))[0]

        # --- Process Data ---
        print("🔧 Processing image and making prediction...")
        pipeline_steps, final_image = self.get_preprocessing_steps(image)
        augmentation_examples = self.get_augmentation_examples(final_image)
        predicted_class, confidence, all_probs = self.make_prediction(final_image)
        print("✅ Processing complete.")

        # --- 1. Create and Save Preprocessing Visual ---
        fig1, axes1 = plt.subplots(1, len(pipeline_steps), figsize=(20, 5)) 
        fig1.suptitle('Preprocessing Steps', fontsize=18, fontweight='bold', y=0.98)
        
        for i, step in enumerate(pipeline_steps):
            ax = axes1[i]
            display_img = step['image']
            
            # Proper normalization for display
            if i == 0:  # Original image (0-255 range)
                display_img = tf.cast(display_img, tf.float32) / 255.0
            elif i == 1:  # Resized image (0-255 range)  
                display_img = tf.cast(display_img, tf.float32) / 255.0
            else:  # Already normalized images (0-1 range, but may exceed due to contrast)
                display_img = tf.cast(display_img, tf.float32)
                # Normalize for display if values exceed [0,1]
                if tf.reduce_max(display_img) > 1.0:
                    display_img = tf.clip_by_value(display_img, 0, 2.0)  # Allow some headroom
                    display_img = display_img / tf.reduce_max(display_img)  # Normalize to [0,1]
            
            # Final safety clipping
            display_img = tf.clip_by_value(display_img, 0, 1)
            ax.imshow(display_img.numpy())
            ax.set_title(step['title'], fontweight='bold', fontsize=12)
            ax.axis('off')
            
        fig1.tight_layout(rect=[0, 0, 1, 0.94])
        self.save_visual(fig1, "report_fig1_preprocessing", base_image_name)

        # --- 2. Create and Save Augmentation Visual ---
        fig2, axes2 = plt.subplots(1, len(augmentation_examples), figsize=(16, 4))
        fig2.suptitle('Data Augmentation Techniques', fontsize=16, fontweight='bold')
        for i, aug in enumerate(augmentation_examples):
            ax = axes2[i]
            img = tf.clip_by_value(aug['image'], 0, 1)
            ax.imshow(img)
            ax.set_title(aug['title'], fontweight='bold')
            ax.axis('off')
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        self.save_visual(fig2, "report_fig2_augmentation", base_image_name)

        # --- 3. Create and Save Prediction Visual ---
        fig3 = plt.figure(figsize=(16, 6))
        fig3.suptitle(f'Prediction Results for g1.jpg', fontsize=18, fontweight='bold', y=0.95)
        
        # Input Image
        ax_img = fig3.add_subplot(1, 3, 1)
        display_final = tf.clip_by_value(final_image, 0, 1)
        ax_img.imshow(display_final.numpy())
        ax_img.set_title('Preprocessed Input\n(128x128, Normalized)', fontweight='bold', fontsize=12)
        ax_img.axis('off')

        # Prediction Text with color coding
        ax_text = fig3.add_subplot(1, 3, 2)
        expected_class = "glioma"  # We know g1.jpg should be glioma
        is_correct = predicted_class.lower() == expected_class.lower()
        
        pred_text = f"PREDICTED:\n{predicted_class.upper()}\n\nEXPECTED:\n{expected_class.upper()}\n\nCONFIDENCE:\n{confidence:.1f}%"
        
        # Color based on correctness
        box_color = 'lightgreen' if is_correct else 'lightcoral'
        status_text = "✅ CORRECT" if is_correct else "❌ WRONG"
        
        ax_text.text(0.5, 0.6, pred_text, ha='center', va='center', fontsize=13, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor=box_color, alpha=0.8))
        ax_text.text(0.5, 0.2, status_text, ha='center', va='center', fontsize=12, fontweight='bold')
        ax_text.axis('off')

        # Enhanced Bar Chart
        ax_bars = fig3.add_subplot(1, 3, 3)
        classes = list(all_probs.keys())
        probs = list(all_probs.values())
        colors = ['darkgreen' if name == predicted_class else 'lightblue' for name in classes]
        
        bars = ax_bars.bar(classes, probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax_bars.set_ylabel('Confidence (%)', fontweight='bold', fontsize=12)
        ax_bars.set_title('Class Probabilities', fontweight='bold', fontsize=12)
        ax_bars.set_ylim(0, 100)
        ax_bars.grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax_bars.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.setp(ax_bars.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fig3.tight_layout(rect=[0, 0, 1, 0.92])
        self.save_visual(fig3, "report_fig3_prediction", base_image_name)

def main():
    """Main function to run the visualizer."""
    print("🧠 Brain Tumor Classification - Report Visual Generator")
    print("=" * 70)
    
    image_path = f"test_image/{IMAGE_NAME}"
    
    print(f"🎯 Generating visuals for image: {image_path}")
    print("-" * 70)
    
    visualizer = ReportVisualizer()
    visualizer.generate_visuals(image_path)
    
    print("\n✅ All report visuals have been generated and saved successfully!")

if __name__ == "__main__":
    main()
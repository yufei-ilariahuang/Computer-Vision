import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

class ImageStitcherGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Stitcher")
        self.root.geometry("400x300")
        
        # Store selected images
        self.selected_images = []
        self.stitcher = ImageStitcher()
        
        # Create GUI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="Image Stitcher", font=("Arial", 16))
        title.pack(pady=20)
        
        # Mode selection
        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=10)
        
        mode_label = tk.Label(mode_frame, text="Select Mode:")
        mode_label.pack()
        
        # Mode buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="2x2 Grid (4 images)", 
                 command=lambda: self.select_images(4)).pack(pady=5)
        tk.Button(btn_frame, text="3x3 Grid (9 images)", 
                 command=lambda: self.select_images(9)).pack(pady=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack(pady=10)
        
        # Stitch button
        self.stitch_btn = tk.Button(self.root, text="Stitch Images", 
                                  command=self.stitch_images, state='disabled')
        self.stitch_btn.pack(pady=10)
        
        # Clear button
        tk.Button(self.root, text="Clear Selection", 
                 command=self.clear_selection).pack(pady=5)
    
    def select_images(self, num_images):
        self.selected_images = []
        files = filedialog.askopenfilenames(
            title=f'Select {num_images} Images',
            filetypes=[
                ('Image files', '*.jpg *.jpeg *.png *.bmp'),
                ('All files', '*.*')
            ]
        )
        
        if len(files) != num_images:
            messagebox.showerror(
                "Error", 
                f"Please select exactly {num_images} images. You selected {len(files)}."
            )
            return
        
        # Load and store the images
        for file in files:
            img = cv2.imread(file)
            if img is not None:
                self.selected_images.append(img)
        
        # Update status
        self.status_label.config(
            text=f"Selected {len(self.selected_images)} images"
        )
        
        # Enable stitch button if correct number of images selected
        if len(self.selected_images) in [4, 9]:
            self.stitch_btn.config(state='normal')
    
    def clear_selection(self):
        self.selected_images = []
        self.status_label.config(text="")
        self.stitch_btn.config(state='disabled')
    
    def stitch_images(self):
        if not self.selected_images:
            messagebox.showerror("Error", "No images selected")
            return
        
        try:
            # Show progress
            self.status_label.config(text="Stitching images... Please wait")
            self.root.update()
            
            # Perform stitching
            result = self.stitcher.stitch_images(self.selected_images)
            
            # Save result
            save_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            
            if save_path:
                cv2.imwrite(save_path, result)
                messagebox.showinfo("Success", "Stitching completed successfully!")
                
                # Display result
                cv2.imshow("Stitched Result", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
        except Exception as e:
            messagebox.showerror("Error", f"Stitching failed: {str(e)}")
        finally:
            self.status_label.config(text="")
    
    def run(self):
        self.root.mainloop()

import argparse
import os
import cv2

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Stitch images from a specified folder.")
    parser.add_argument("folder_path", help="Path to the folder containing images to stitch")
    args = parser.parse_args()

    
    # Read images from the specified folder
    images = []
    for filename in sorted(os.listdir(args.folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(args.folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                print(f"Loaded image: {filename}")
    
    if not images:
        print("No valid images found in the specified folder.")
        return

    # Stitch images
    try:
        print(f"\nStarting stitching process with {len(images)} images...")
        result = stitcher.stitch_images(images)
        
        # Save and display result
        output_path = os.path.join(args.folder_path, "stitched_result.jpg")
        cv2.imwrite(output_path, result)
        print(f"\nResult saved successfully at: {output_path}")
        
        cv2.imshow("Stitched Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during stitching: {str(e)}")

if __name__ == "__main__":
    main()
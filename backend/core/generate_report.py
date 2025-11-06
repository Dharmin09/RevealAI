from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
from datetime import datetime
import numpy as np

class SimpleReport:
    def __init__(self, out_path="report.pdf"):
        self.pdf = FPDF()
        self.out_path = out_path
        self.pdf.set_auto_page_break(auto=True, margin=12)
        self.pdf.set_font("Arial", size=11)

    def add_cover(self, title="RevealAI - Detection Report"):
        """Create a professional cover page with black/neon cyan theme"""
        self.pdf.add_page()
        
        # Main background - Black
        self.pdf.set_fill_color(15, 15, 30)  # Deep black/dark navy
        self.pdf.rect(0, 0, 210, 297, 'F')
        
        # Neon cyan accent bar at top
        self.pdf.set_fill_color(0, 255, 255)  # Neon cyan
        self.pdf.rect(0, 0, 210, 8, 'F')
        
        # Neon cyan accent bar near bottom of header
        self.pdf.set_fill_color(0, 255, 255)  # Neon cyan
        self.pdf.rect(0, 85, 210, 3, 'F')
        
        # Title section - RevealAI
        self.pdf.set_text_color(0, 255, 255)  # Neon cyan text
        self.pdf.set_font("Arial", 'B', size=48)
        self.pdf.ln(25)
        self.pdf.cell(0, 25, "RevealAI", ln=True, align='C')
        
        # Subtitle
        self.pdf.set_text_color(100, 255, 255)  # Lighter cyan
        self.pdf.set_font("Arial", size=16)
        self.pdf.cell(0, 8, "Professional Deepfake Detection Report", ln=True, align='C')
        
        # Spacer
        self.pdf.ln(15)
        
        # Report Information section
        self.pdf.set_text_color(255, 255, 255)  # White text
        self.pdf.set_font("Arial", 'B', size=12)
        self.pdf.cell(0, 8, "Report Information", ln=True)
        
        # Separator line in neon cyan
        self.pdf.set_draw_color(0, 255, 255)
        self.pdf.set_line_width(0.5)
        current_x = self.pdf.get_x()
        current_y = self.pdf.get_y()
        self.pdf.line(15, current_y, 195, current_y)
        self.pdf.ln(4)
        
        # Report details
        self.pdf.set_font("Arial", size=11)
        report_date = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        
        self.pdf.cell(40, 7, "Generated:", 0, 0)
        self.pdf.set_text_color(0, 255, 255)  # Neon cyan for values
        self.pdf.cell(0, 7, report_date, 0, 1)
        
        self.pdf.set_text_color(255, 255, 255)  # White for labels
        self.pdf.cell(40, 7, "System:", 0, 0)
        self.pdf.set_text_color(0, 255, 255)  # Neon cyan for values
        self.pdf.cell(0, 7, "RevealAI v2.0", 0, 1)
        
        self.pdf.set_text_color(255, 255, 255)  # White for labels
        self.pdf.cell(40, 7, "Analysis Type:", 0, 0)
        self.pdf.set_text_color(0, 255, 255)  # Neon cyan for values
        self.pdf.cell(0, 7, "Multi-Modal Detection", 0, 1)
        
        # Reset colors for next page
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_draw_color(0, 0, 0)

    def add_result(self, filename, video_score, audio_score, final_score, heatmaps=[], original_frames=[], spec_img=None, audio_heatmap=None, verdict="uncertain", findings=None):
        """
        Add comprehensive analysis result page
        
        Args:
            filename: Name of analyzed file
            video_score: Video analysis score (0.0-1.0)
            audio_score: Audio analysis score (0.0-1.0)
            final_score: Combined score (0.0-1.0)
            heatmaps: List of heatmap numpy arrays
            original_frames: List of original frame numpy arrays (to display side-by-side)
            spec_img: Spectrogram numpy array (small, for model input visualization)
            audio_heatmap: Audio heatmap visualization (large professional heatmap for display)
            verdict: Detection verdict string
            findings: Dict with 'summary' and 'details' (list of findings)
        """
        # --- Page 1: Summary & Verdict ---
        self.pdf.add_page()
        
        # Add neon cyan top line accent
        self.pdf.set_draw_color(0, 255, 255)
        self.pdf.set_line_width(1)
        self.pdf.line(10, 10, 200, 10)
        self.pdf.ln(3)
        
        self.pdf.set_font("Arial", 'B', size=18)
        self.pdf.set_text_color(0, 100, 100)  # Cyan color
        self.pdf.cell(0, 10, "Analysis Summary", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(2)
        
        # Determine verdict color and description
        if final_score > 0.7:
            verdict_text = "LIKELY DEEPFAKE"
            verdict_color = (192, 57, 43)  # Red
            confidence = "High"
        elif final_score > 0.5:
            verdict_text = "SUSPICIOUS"
            verdict_color = (230, 126, 34)  # Orange
            confidence = "Medium"
        elif final_score > 0.3:
            verdict_text = "UNCERTAIN"
            verdict_color = (155, 89, 182)  # Purple
            confidence = "Low"
        else:
            verdict_text = "LIKELY AUTHENTIC"
            verdict_color = (46, 204, 113)  # Green
            confidence = "High"
        
        # Verdict box with color and border
        self.pdf.set_fill_color(verdict_color[0], verdict_color[1], verdict_color[2])
        self.pdf.set_draw_color(0, 0, 0)
        self.pdf.set_line_width(0.5)
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.set_font("Arial", 'B', size=16)
        self.pdf.cell(0, 15, verdict_text, ln=True, align='C', fill=True)
        self.pdf.set_text_color(0, 0, 0)
        
        self.pdf.ln(5)
        self.pdf.set_font("Arial", 'B', size=12)
        self.pdf.set_text_color(0, 100, 100)  # Cyan color for section headers
        self.pdf.cell(0, 8, "File Information", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_font("Arial", size=11)
        self.pdf.cell(50, 7, "Filename:", 0, 0)
        self.pdf.cell(0, 7, str(filename)[:60], 0, 1)
        
        self.pdf.ln(6)
        self.pdf.set_font("Arial", 'B', size=12)
        self.pdf.set_text_color(0, 100, 100)  # Cyan color
        self.pdf.cell(0, 8, "Detection Scores", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_font("Arial", size=11)
        
        # Score table with better formatting
        self.pdf.cell(70, 7, "Video Analysis Score:", 0, 0)
        self.pdf.set_text_color(0, 150, 150)  # Cyan for values
        self.pdf.cell(50, 7, f"{video_score:.1%}", 0, 1)
        self.pdf.set_text_color(0, 0, 0)
        
        self.pdf.cell(70, 7, "Audio Analysis Score:", 0, 0)
        self.pdf.set_text_color(0, 150, 150)  # Cyan for values
        self.pdf.cell(50, 7, f"{audio_score:.1%}", 0, 1)
        self.pdf.set_text_color(0, 0, 0)
        
        self.pdf.set_font("Arial", 'B', size=11)
        self.pdf.cell(70, 8, "FINAL CONFIDENCE SCORE:", 0, 0)
        self.pdf.set_text_color(255, 0, 0)  # Red for final score
        self.pdf.cell(50, 8, f"{final_score:.1%}", 0, 1)
        self.pdf.set_text_color(0, 0, 0)
        
        self.pdf.set_font("Arial", size=11)
        self.pdf.cell(70, 7, "Detection Confidence:", 0, 0)
        self.pdf.set_text_color(0, 150, 150)  # Cyan for values
        self.pdf.cell(50, 7, confidence, 0, 1)
        self.pdf.set_text_color(0, 0, 0)
        
        self.pdf.ln(5)
        self.pdf.set_font("Arial", 'B', size=13)
        self.pdf.cell(0, 8, "How to Interpret These Results", ln=True)
        self.pdf.set_font("Arial", size=10)
        
        # Use simple text instead of multi_cell to avoid Unicode issues
        self.pdf.cell(0, 5, "Score Range:", ln=True)
        self.pdf.cell(5, 5, "")
        self.pdf.cell(0, 5, "0.0 - 0.3: Likely AUTHENTIC (genuine content)", ln=True)
        self.pdf.cell(5, 5, "")
        self.pdf.cell(0, 5, "0.3 - 0.5: UNCERTAIN (requires further investigation)", ln=True)
        self.pdf.cell(5, 5, "")
        self.pdf.cell(0, 5, "0.5 - 0.7: SUSPICIOUS (possible deepfake indicators detected)", ln=True)
        self.pdf.cell(5, 5, "")
        self.pdf.cell(0, 5, "0.7 - 1.0: Likely DEEPFAKE (strong evidence of synthesis)", ln=True)
        self.pdf.ln(3)
        self.pdf.set_font("Arial", size=9)
        self.pdf.multi_cell(0, 4,
            "The AI model was trained to detect artifacts common in AI-generated or manipulated media, including compression patterns, lighting inconsistencies, and unnatural facial movements.")
        
        # --- Page 2: What We Found (Detailed Findings) ---
        if findings and isinstance(findings, dict):
            self.pdf.add_page()
            self.pdf.set_font("Arial", 'B', size=16)
            self.pdf.cell(0, 10, "What We Found", ln=True)
            self.pdf.ln(3)
            
            # Main summary in a highlighted box
            if 'explanation' in findings:
                self.pdf.set_fill_color(240, 240, 240)  # Light gray background
                self.pdf.set_font("Arial", size=10)
                self.pdf.cell(5, 5, "")
                self.pdf.multi_cell(0, 5, findings['explanation'], fill=True)
                self.pdf.ln(3)
            
            # Detailed findings list
            if 'findings' in findings and isinstance(findings['findings'], list):
                self.pdf.set_font("Arial", 'B', size=11)
                self.pdf.cell(0, 8, "Key Findings:", ln=True)
                self.pdf.ln(2)
                
                self.pdf.set_font("Arial", size=10)
                for i, finding in enumerate(findings['findings'], 1):
                    # Use bullet character (dash) instead of special chars
                    finding_text = str(finding).strip()
                    
                    # Add indentation and dash
                    self.pdf.set_x(20)
                    self.pdf.multi_cell(0, 5, f"- {finding_text}", 
                                       split_only=False)
                    self.pdf.ln(1)
        
        # --- Page 3: Visual Evidence (Heatmaps with Originals) ---
        if heatmaps and len(heatmaps) > 0:
            self.pdf.add_page()
            
            # Add neon cyan accent line at top
            self.pdf.set_draw_color(0, 255, 255)
            self.pdf.set_line_width(1)
            self.pdf.line(10, 10, 200, 10)
            
            self.pdf.set_font("Arial", 'B', size=16)
            self.pdf.set_text_color(0, 100, 100)  # Cyan-ish color
            self.pdf.cell(0, 10, "Frame Analysis with Heatmap Overlay", ln=True)
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.ln(2)
            
            self.pdf.set_font("Arial", size=9)
            self.pdf.multi_cell(0, 4,
                "Below are side-by-side comparisons of original frames and AI detection heatmaps. Left: Original frame from video. Right: AI model heatmap showing suspicious regions (red=high suspicion, blue=authentic areas).")
            self.pdf.ln(2)
            
            # Display heatmaps with originals side-by-side
            img_width = 75  # Slightly smaller to fit verdict below
            spacing = 8     # Space between columns
            
            for i, img_arr in enumerate(heatmaps):
                try:
                    # Check if we need page break
                    if self.pdf.get_y() > 240:
                        self.pdf.add_page()
                        self.pdf.ln(5)
                    
                    # Frame label with neon cyan styling
                    self.pdf.set_font("Arial", 'B', size=11)
                    self.pdf.set_text_color(0, 150, 150)  # Cyan color
                    self.pdf.cell(0, 7, f"Frame {i+1}", ln=True)
                    self.pdf.set_text_color(0, 0, 0)
                    
                    # Column headers
                    self.pdf.set_font("Arial", 'B', size=8)
                    self.pdf.cell(img_width, 5, "Original Frame", align='C')
                    self.pdf.cell(spacing, 5, "")
                    self.pdf.cell(img_width, 5, "AI Detection Heatmap", align='C', ln=True)
                    
                    # Get Y position for images
                    img_y_start = self.pdf.get_y()
                    
                    # Prepare heatmap image
                    tmp_hm = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tmp_hm_path = tmp_hm.name
                    tmp_hm.close()
                    
                    # Convert heatmap to PIL and save
                    if isinstance(img_arr, np.ndarray):
                        if img_arr.dtype != np.uint8:
                            hm_img_uint8 = (img_arr * 255).astype(np.uint8) if img_arr.max() <= 1 else img_arr.astype(np.uint8)
                        else:
                            hm_img_uint8 = img_arr
                        Image.fromarray(hm_img_uint8).save(tmp_hm_path)
                    else:
                        img_arr.save(tmp_hm_path)
                    
                    # Display original frame if available (use placeholder if not)
                    if original_frames and i < len(original_frames) and original_frames[i] is not None:
                        tmp_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        tmp_orig_path = tmp_orig.name
                        tmp_orig.close()
                        
                        orig_img = original_frames[i]
                        if isinstance(orig_img, np.ndarray):
                            if orig_img.dtype != np.uint8:
                                orig_img_uint8 = (orig_img * 255).astype(np.uint8) if orig_img.max() <= 1 else orig_img.astype(np.uint8)
                            else:
                                orig_img_uint8 = orig_img
                            Image.fromarray(orig_img_uint8).save(tmp_orig_path)
                        else:
                            orig_img.save(tmp_orig_path)
                        
                        # Add original frame image with border
                        self.pdf.set_draw_color(0, 255, 255)  # Neon cyan border
                        self.pdf.set_line_width(0.3)
                        self.pdf.image(tmp_orig_path, x=10, y=img_y_start, w=img_width, h=img_width)
                    else:
                        # Draw placeholder for original frame
                        self.pdf.set_xy(10, img_y_start)
                        self.pdf.set_draw_color(100, 100, 100)
                        self.pdf.rect(10, img_y_start, img_width, img_width)
                        self.pdf.set_font("Arial", size=8)
                        self.pdf.set_xy(10, img_y_start + img_width//2 - 2)
                        self.pdf.cell(img_width, 4, "(Original frame)", align='C')
                    
                    # Add heatmap image on right with border
                    self.pdf.set_draw_color(0, 255, 255)  # Neon cyan border
                    self.pdf.set_line_width(0.3)
                    self.pdf.image(tmp_hm_path, x=10 + img_width + spacing, y=img_y_start, w=img_width, h=img_width)
                    
                    # Move below images
                    self.pdf.set_y(img_y_start + img_width + 2)
                    
                    # Add verdict below frames
                    self.pdf.set_font("Arial", 'B', size=9)
                    verdict_text = f"Verdict: {verdict.upper()}"
                    
                    # Color code the verdict
                    if "DEEPFAKE" in verdict.upper():
                        self.pdf.set_text_color(255, 0, 0)  # Red
                    elif "SUSPICIOUS" in verdict.upper():
                        self.pdf.set_text_color(255, 165, 0)  # Orange
                    elif "UNCERTAIN" in verdict.upper():
                        self.pdf.set_text_color(155, 89, 182)  # Purple
                    else:
                        self.pdf.set_text_color(0, 200, 0)  # Green
                    
                    self.pdf.cell(0, 5, verdict_text, ln=True, align='C')
                    self.pdf.set_text_color(0, 0, 0)
                    self.pdf.ln(2)
                    
                    # Cleanup temp files
                    if os.path.exists(tmp_hm_path):
                        try:
                            os.unlink(tmp_hm_path)
                        except:
                            pass
                    if 'tmp_orig_path' in locals() and os.path.exists(tmp_orig_path):
                        try:
                            os.unlink(tmp_orig_path)
                        except:
                            pass
                            
                except Exception as e:
                    print(f"[WARN] Could not add frame {i+1}: {e}")
                    self.pdf.set_font("Arial", size=9)
                    self.pdf.cell(0, 6, f"Error displaying Frame {i+1}", ln=True)
        
        # --- Page 4: Audio Analysis (Spectrogram & Heatmap) ---
        if spec_img is not None or audio_heatmap is not None:
            try:
                self.pdf.add_page()
                
                # Add neon cyan accent line at top
                self.pdf.set_draw_color(0, 255, 255)
                self.pdf.set_line_width(1)
                self.pdf.line(10, 10, 200, 10)
                
                self.pdf.set_font("Arial", 'B', size=16)
                self.pdf.set_text_color(0, 100, 100)  # Cyan-ish color
                self.pdf.cell(0, 10, "Audio Analysis", ln=True)
                self.pdf.set_text_color(0, 0, 0)
                self.pdf.ln(3)
                
                # Explanation
                self.pdf.set_font("Arial", size=9)
                if audio_heatmap is not None:
                    self.pdf.multi_cell(0, 4,
                        "Spectrogram Heatmap Analysis: The visualization below is a professional audio spectrogram heatmap showing frequency content (vertical axis) over time (horizontal axis). Red/hot colors indicate high energy regions that the AI model analyzes for unnatural patterns, artifacts, and synthesis indicators of AI-generated or manipulated speech.")
                else:
                    self.pdf.multi_cell(0, 4,
                        "Spectrogram Analysis: The image below is a spectrogram representation of the audio signal. It shows frequency content over time. The AI model analyzes this for unnatural patterns and artifacts that indicate synthetic or manipulated speech.")
                self.pdf.ln(3)
                
                # Display audio heatmap if available (preferred for visual quality)
                if audio_heatmap is not None:
                    try:
                        tmp_hm = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        tmp_hm_path = tmp_hm.name
                        tmp_hm.close()
                        
                        # Ensure audio_heatmap is uint8 numpy array with correct shape
                        if isinstance(audio_heatmap, np.ndarray):
                            # Make a copy to avoid modifying original
                            hm_copy = audio_heatmap.copy()
                            
                            # Ensure it's uint8
                            if hm_copy.dtype != np.uint8:
                                if hm_copy.max() <= 1.0:
                                    hm_uint8 = (hm_copy * 255).astype(np.uint8)
                                else:
                                    hm_uint8 = hm_copy.astype(np.uint8)
                            else:
                                hm_uint8 = hm_copy
                            
                            # Ensure shape is (height, width, 3) for RGB
                            if len(hm_uint8.shape) == 2:
                                # Grayscale - convert to RGB
                                hm_uint8 = np.stack([hm_uint8] * 3, axis=-1)
                            elif len(hm_uint8.shape) == 3 and hm_uint8.shape[2] == 4:
                                # RGBA - convert to RGB
                                hm_uint8 = hm_uint8[:, :, :3]
                            
                            print(f"[REPORT] Audio heatmap shape: {hm_uint8.shape}, dtype: {hm_uint8.dtype}, min: {hm_uint8.min()}, max: {hm_uint8.max()}")
                            
                            # Save as PIL Image
                            pil_img = Image.fromarray(hm_uint8, mode='RGB')
                            pil_img.save(tmp_hm_path, format='PNG')
                        else:
                            # Already a PIL Image
                            audio_heatmap.save(tmp_hm_path, format='PNG')
                        
                        # Verify file was created
                        if not os.path.exists(tmp_hm_path):
                            raise Exception("Failed to create temporary heatmap file")
                        
                        # Add large professional heatmap (scales to fit page)
                        self.pdf.image(tmp_hm_path, x=10, w=190)
                        self.pdf.ln(2)
                        
                        # Add interpretation text
                        self.pdf.set_font("Arial", size=8)
                        self.pdf.set_text_color(100, 100, 100)  # Gray text
                        self.pdf.multi_cell(0, 3,
                            "Legend: Dark/Blue regions = Low energy (authentic pauses/silence) | Yellow/Red regions = High energy (speech content). AI-generated speech often shows unnatural frequency distributions and inconsistent energy patterns.")
                        self.pdf.set_text_color(0, 0, 0)
                        
                        # Cleanup
                        if os.path.exists(tmp_hm_path):
                            try:
                                os.unlink(tmp_hm_path)
                            except:
                                pass
                    
                    except Exception as e:
                        print(f"[WARN] Could not add audio heatmap: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to small spectrogram if heatmap fails
                        if spec_img is not None:
                            self._add_spectrogram_fallback(spec_img)
                
                # Display small spectrogram as fallback or secondary view (only if we couldn't show heatmap)
                elif spec_img is not None:
                    self._add_spectrogram_fallback(spec_img)
                
            except Exception as e:
                print(f"[WARN] Could not add audio analysis: {e}")

    def _add_spectrogram_fallback(self, spec_img):
        """Helper function to add small spectrogram when heatmap is not available"""
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp_path = tmp.name
            tmp.close()
            
            # Convert to PIL and save
            if isinstance(spec_img, np.ndarray):
                if spec_img.dtype != np.uint8:
                    spec_img_uint8 = (spec_img * 255).astype(np.uint8) if spec_img.max() <= 1 else spec_img.astype(np.uint8)
                else:
                    spec_img_uint8 = spec_img
                Image.fromarray(spec_img_uint8).save(tmp_path)
            else:
                spec_img.save(tmp_path)
            
            # Center and add spectrogram
            self.pdf.image(tmp_path, x=15, w=180)
            
        except Exception as e:
            print(f"[WARN] Could not add spectrogram fallback: {e}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    def output(self):
        """Output PDF file"""
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        self.pdf.output(self.out_path)
        return self.out_path

def generate_report_for_media(media_type, filename, analysis_results, video_metadata=None, audio_metadata=None):
    """
    High-level function to generate a complete PDF report for video or audio.

    Args:
        media_type (str): 'video' or 'audio'.
        filename (str): The name of the analyzed file.
        analysis_results (dict): The dictionary returned from infer_video or infer_audio.
        video_metadata (dict, optional): Extracted video metadata.
        audio_metadata (dict, optional): Extracted audio metadata.

    Returns:
        str: The path to the generated PDF report.
    """
    # Create a temporary file for the report
    temp_report_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix=f"report_{media_type}_")
    report_path = temp_report_file.name
    temp_report_file.close()

    report = SimpleReport(out_path=report_path)
    report.add_cover()

    # Extract data from analysis results
    video_score = analysis_results.get('video_score', 0.0)
    audio_score = analysis_results.get('audio_score', 0.0)
    heatmaps = analysis_results.get('heatmaps', [])
    original_frames = analysis_results.get('original_frames', [])
    spec_img = analysis_results.get('spec_img')
    audio_heatmap = analysis_results.get('audio_heatmap')

    # Combine scores (60% video, 40% audio)
    if media_type == 'video':
        final_score = video_score
    elif media_type == 'audio':
        final_score = audio_score
    else: # Combined
        final_score = (video_score * 0.6) + (audio_score * 0.4)

    # Determine verdict
    if final_score > 0.7:
        verdict = "Likely Deepfake"
    elif final_score > 0.4:
        verdict = "Suspicious"
    else:
        verdict = "Likely Authentic"

    report.add_result(
        filename=filename,
        video_score=video_score,
        audio_score=audio_score,
        final_score=final_score,
        heatmaps=heatmaps,
        original_frames=original_frames,
        spec_img=spec_img,
        audio_heatmap=audio_heatmap,
        verdict=verdict,
        findings={'explanation': f"The {media_type} analysis resulted in a score of {final_score:.2f}."}
    )

    return report.output()

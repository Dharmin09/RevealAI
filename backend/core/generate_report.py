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
        """Create a professional cover page with white background and black text"""
        self.pdf.add_page()
        # White background (default)
        self.pdf.set_fill_color(255, 255, 255)
        self.pdf.rect(0, 0, 210, 297, 'F')
        # Title section
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_font("Arial", 'B', size=44)
        self.pdf.ln(30)
        self.pdf.cell(0, 25, "RevealAI", ln=True, align='C')
        # Subtitle
        self.pdf.set_font("Arial", size=16)
        self.pdf.cell(0, 8, "Professional Deepfake Detection Report", ln=True, align='C')
        self.pdf.ln(18)
        # Report Information section
        self.pdf.set_font("Arial", 'B', size=12)
        self.pdf.cell(0, 8, "Report Information", ln=True)
        self.pdf.set_draw_color(0, 0, 0)
        self.pdf.set_line_width(0.5)
        current_y = self.pdf.get_y()
        self.pdf.line(15, current_y, 195, current_y)
        self.pdf.ln(4)
        # Report details
        self.pdf.set_font("Arial", size=11)
        report_date = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(40, 7, "Generated:", 0, 0)
        self.pdf.cell(0, 7, report_date, 0, 1)
        self.pdf.cell(40, 7, "System:", 0, 0)
        self.pdf.cell(0, 7, "RevealAI v2.0", 0, 1)
        self.pdf.cell(40, 7, "Analysis Type:", 0, 0)
        self.pdf.cell(0, 7, "Multi-Modal Detection", 0, 1)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_draw_color(0, 0, 0)

    def add_result(self, filename, video_score, audio_score, final_score, heatmaps=None, original_frames=None, spec_img=None, audio_heatmap=None, verdict="uncertain", findings=None, video_metadata=None, audio_metadata=None, show_video_score=True, show_audio_score=True):
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
            video_metadata: Optional dict of video metadata details
            audio_metadata: Optional dict of audio metadata details
        """
        heatmaps = heatmaps or []
        original_frames = original_frames or []
        video_metadata = video_metadata or {}
        audio_metadata = audio_metadata or {}

        # --- Page 1: Summary & Verdict ---
        self.pdf.add_page()
        # Add black top line accent
        self.pdf.set_draw_color(0, 0, 0)
        self.pdf.set_line_width(1)
        self.pdf.line(10, 10, 200, 10)
        self.pdf.ln(3)
        self.pdf.set_font("Arial", 'B', size=18)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(0, 10, "Analysis Summary", ln=True)
        self.pdf.ln(2)
        
        # Determine verdict color and description (red/green/yellow only)
        if final_score > 0.7:
            verdict_text = "LIKELY DEEPFAKE"
            verdict_color = (255, 0, 0)  # Red
            text_color = (180, 0, 0)
            confidence = "High"
        elif final_score > 0.5:
            verdict_text = "SUSPICIOUS"
            verdict_color = (255, 215, 0)  # Yellow
            text_color = (180, 120, 0)
            confidence = "Medium"
        elif final_score > 0.3:
            verdict_text = "UNCERTAIN"
            verdict_color = (200, 200, 200)  # Gray
            text_color = (80, 80, 80)
            confidence = "Low"
        else:
            verdict_text = "LIKELY AUTHENTIC"
            verdict_color = (0, 180, 0)  # Green
            text_color = (0, 120, 0)
            confidence = "High"
        # Verdict box with border only, white fill
        self.pdf.set_fill_color(255, 255, 255)
        self.pdf.set_draw_color(*verdict_color)
        self.pdf.set_line_width(1)
        self.pdf.set_xy(55, self.pdf.get_y())
        self.pdf.cell(100, 15, '', border=1, ln=1, align='C', fill=True)
        self.pdf.set_xy(0, self.pdf.get_y() - 15)
        self.pdf.set_text_color(*text_color)
        self.pdf.set_font("Arial", 'B', size=16)
        self.pdf.cell(0, 15, verdict_text, ln=True, align='C')
        self.pdf.set_text_color(0, 0, 0)
        
        self.pdf.ln(5)
        self.pdf.set_font("Arial", 'B', size=12)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(0, 8, "File Information", ln=True)
        self.pdf.set_font("Arial", size=11)
        self.pdf.cell(50, 7, "Filename:", 0, 0)
        self.pdf.cell(0, 7, str(filename)[:60], 0, 1)
        
        # Include metadata tables when provided
        def _render_metadata_section(title, metadata_dict):
            if not metadata_dict:
                return
            self.pdf.ln(4)
            self.pdf.set_font("Arial", 'B', size=11)
            self.pdf.set_text_color(80, 80, 80)  # Gray for section headers
            self.pdf.cell(0, 7, title, ln=True)
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.set_font("Arial", size=10)
            for key, value in metadata_dict.items():
                label = str(key).replace('_', ' ').title()
                self.pdf.cell(55, 6, f"{label}:", 0, 0)
                self.pdf.cell(0, 6, str(value), 0, 1)

        _render_metadata_section("Video Metadata", video_metadata)
        _render_metadata_section("Audio Metadata", audio_metadata)

        self.pdf.ln(6)
        self.pdf.set_font("Arial", 'B', size=12)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(0, 8, "Detection Scores", ln=True)
        self.pdf.set_font("Arial", size=11)
        # Score table with better formatting
        self.pdf.cell(70, 7, "Video Analysis Score:", 0, 0)
        self.pdf.set_text_color(0, 0, 0)
        if show_video_score:
            self.pdf.cell(50, 7, f"{video_score:.1%}", 0, 1)
        else:
            self.pdf.cell(50, 7, "N/A", 0, 1)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(70, 7, "Audio Analysis Score:", 0, 0)
        self.pdf.set_text_color(0, 0, 0)
        if show_audio_score:
            self.pdf.cell(50, 7, f"{audio_score:.1%}", 0, 1)
        else:
            self.pdf.cell(50, 7, "N/A", 0, 1)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_font("Arial", 'B', size=11)
        self.pdf.cell(70, 8, "FINAL CONFIDENCE SCORE:", 0, 0)
        # Red for high, yellow for medium, green for low
        if final_score > 0.7:
            score_color = (255, 0, 0)
        elif final_score > 0.5:
            score_color = (255, 215, 0)
        elif final_score > 0.3:
            score_color = (80, 80, 80)
        else:
            score_color = (0, 180, 0)
        self.pdf.set_text_color(*score_color)
        self.pdf.cell(50, 8, f"{final_score:.1%}", 0, 1)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_font("Arial", size=11)
        self.pdf.cell(70, 7, "Detection Confidence:", 0, 0)
        self.pdf.set_text_color(80, 80, 80)
        self.pdf.cell(50, 7, confidence, 0, 1)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(5)
        self.pdf.set_font("Arial", 'B', size=13)
        self.pdf.cell(0, 8, "How to Interpret These Results", ln=True)
        self.pdf.set_font("Arial", size=10)
        self.pdf.cell(0, 5, "Score Range:", ln=True)
        self.pdf.cell(5, 5, "")
        self.pdf.set_text_color(0, 180, 0)
        self.pdf.cell(0, 5, "0.0 - 0.3: Likely AUTHENTIC (genuine content)", ln=True)
        self.pdf.set_text_color(80, 80, 80)
        self.pdf.cell(5, 5, "")
        self.pdf.cell(0, 5, "0.3 - 0.5: UNCERTAIN (requires further investigation)", ln=True)
        self.pdf.set_text_color(255, 215, 0)
        self.pdf.cell(5, 5, "")
        self.pdf.cell(0, 5, "0.5 - 0.7: SUSPICIOUS (possible deepfake indicators detected)", ln=True)
        self.pdf.set_text_color(255, 0, 0)
        self.pdf.cell(5, 5, "")
        self.pdf.cell(0, 5, "0.7 - 1.0: Likely DEEPFAKE (strong evidence of synthesis)", ln=True)
        self.pdf.set_text_color(0, 0, 0)
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

            # Always show verdict box
            if verdict is not None:
                if verdict.upper() == "REAL" or verdict.upper() == "LIKELY AUTHENTIC":
                    verdict_color = (0, 180, 0)
                    text_color = (0, 120, 0)
                    verdict_text = "LIKELY AUTHENTIC"
                    xai_explanation = "The audio signal is consistent with genuine, unaltered speech. No significant signs of manipulation or synthesis were detected."
                elif verdict.upper() == "FAKE" or verdict.upper() == "LIKELY DEEPFAKE":
                    verdict_color = (255, 0, 0)
                    text_color = (180, 0, 0)
                    verdict_text = "LIKELY DEEPFAKE"
                    xai_explanation = "The AI detected strong evidence of synthetic or manipulated audio, such as unnatural frequency patterns or artifacts typical of deepfake generation."
                elif verdict.upper() == "SUSPICIOUS":
                    verdict_color = (255, 215, 0)
                    text_color = (180, 120, 0)
                    verdict_text = "SUSPICIOUS"
                    xai_explanation = "Some anomalies were found in the audio that may indicate manipulation, but the evidence is not conclusive. Further review is recommended."
                else:
                    verdict_color = (200, 200, 200)
                    text_color = (80, 80, 80)
                    verdict_text = verdict.upper()
                    xai_explanation = "The AI was unable to confidently classify the audio. Please review the analysis details for more information."
                self.pdf.set_fill_color(255, 255, 255)
                self.pdf.set_draw_color(*verdict_color)
                self.pdf.set_line_width(1)
                y = self.pdf.get_y()
                self.pdf.set_xy(55, y)
                self.pdf.cell(100, 12, '', border=1, ln=1, align='C', fill=True)
                self.pdf.set_xy(0, y)
                self.pdf.set_text_color(*text_color)
                self.pdf.set_font("Arial", 'B', size=14)
                self.pdf.cell(0, 12, f"VERDICT: {verdict_text}", ln=True, align='C')
                self.pdf.set_text_color(0, 0, 0)
                self.pdf.ln(2)
                # Add Explainable AI Verdict
                self.pdf.set_font("Arial", 'B', size=11)
                self.pdf.cell(0, 7, "Explainable AI Verdict:", ln=True)
                self.pdf.set_font("Arial", size=10)
                self.pdf.set_fill_color(245, 245, 245)
                self.pdf.multi_cell(0, 5, xai_explanation, fill=True)
                self.pdf.ln(2)

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
                    finding_text = str(finding).strip()
                    self.pdf.set_x(20)
                    self.pdf.multi_cell(0, 5, f"- {finding_text}", split_only=False)
                    self.pdf.ln(1)
        
        # --- Page: Frame Analysis with Heatmap Overlay ---
        if heatmaps and len(heatmaps) > 0:
            self.pdf.add_page()
            self.pdf.set_draw_color(0, 0, 0)
            self.pdf.set_line_width(1)
            self.pdf.line(10, 10, 200, 10)
            self.pdf.set_font("Arial", 'B', size=16)
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.cell(0, 10, "Frame Analysis with Heatmap Overlay", ln=True)
            self.pdf.ln(2)
            self.pdf.set_font("Arial", size=9)
            self.pdf.multi_cell(0, 4,
                "Below are side-by-side comparisons of original frames and AI detection heatmaps. Left: Original frame from video. Right: AI model heatmap showing suspicious regions.")
            self.pdf.ln(2)
            img_width = 75
            spacing = 8
            for i, img_arr in enumerate(heatmaps):
                try:
                    if self.pdf.get_y() > 240:
                        self.pdf.add_page()
                        self.pdf.ln(5)
                    self.pdf.set_font("Arial", 'B', size=11)
                    self.pdf.set_text_color(80, 80, 80)
                    self.pdf.cell(0, 7, f"Frame {i+1}", ln=True)
                    self.pdf.set_text_color(0, 0, 0)
                    self.pdf.set_font("Arial", 'B', size=8)
                    self.pdf.cell(img_width, 5, "Original Frame", align='C')
                    self.pdf.cell(spacing, 5, "")
                    self.pdf.cell(img_width, 5, "AI Detection Heatmap", align='C', ln=True)
                    img_y_start = self.pdf.get_y()
                    # Prepare heatmap image
                    tmp_hm = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    tmp_hm_path = tmp_hm.name
                    tmp_hm.close()
                    if isinstance(img_arr, np.ndarray):
                        if img_arr.dtype != np.uint8:
                            hm_img_uint8 = (img_arr * 255).astype(np.uint8) if img_arr.max() <= 1 else img_arr.astype(np.uint8)
                        else:
                            hm_img_uint8 = img_arr
                        Image.fromarray(hm_img_uint8).save(tmp_hm_path)
                    else:
                        img_arr.save(tmp_hm_path)
                    # Display original frame if available
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
                        self.pdf.set_draw_color(0, 0, 0)
                        self.pdf.set_line_width(0.3)
                        self.pdf.image(tmp_orig_path, x=10, y=img_y_start, w=img_width, h=img_width)
                    else:
                        self.pdf.set_xy(10, img_y_start)
                        self.pdf.set_draw_color(100, 100, 100)
                        self.pdf.rect(10, img_y_start, img_width, img_width)
                        self.pdf.set_font("Arial", size=8)
                        self.pdf.set_xy(10, img_y_start + img_width//2 - 2)
                        self.pdf.cell(img_width, 4, "(Original frame)", align='C')
                    self.pdf.set_draw_color(0, 0, 0)
                    self.pdf.set_line_width(0.3)
                    self.pdf.image(tmp_hm_path, x=10 + img_width + spacing, y=img_y_start, w=img_width, h=img_width)
                    self.pdf.set_y(img_y_start + img_width + 2)
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
                    # Hide all errors in PDF generation
                    pass
        
        # --- Page 4: Audio Analysis (Spectrogram & Heatmap) ---
        # (Removed for audio-only reports)

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
    video_score_raw = analysis_results.get('video_score')
    audio_score_raw = analysis_results.get('audio_score')

    try:
        video_score = float(video_score_raw) if video_score_raw is not None else 0.0
    except (TypeError, ValueError):
        video_score = 0.0
        video_score_raw = None

    try:
        audio_score = float(audio_score_raw) if audio_score_raw is not None else 0.0
    except (TypeError, ValueError):
        audio_score = 0.0
        audio_score_raw = None

    show_video_score = video_score_raw is not None
    show_audio_score = audio_score_raw is not None
    # Only use current upload's data, do not retrieve or merge with previous uploads
    heatmaps = analysis_results.get('heatmaps', [])
    original_frames = analysis_results.get('original_frames', [])
    spec_img = analysis_results.get('spec_img')
    audio_heatmap = analysis_results.get('audio_heatmap')

    # Combine scores (60% video, 40% audio)
    final_score = analysis_results.get('final_score')
    if final_score is None:
        if media_type == 'video':
            final_score = video_score
        elif media_type == 'audio':
            final_score = audio_score
        else:  # Combined
            final_score = (video_score * 0.6) + (audio_score * 0.4)

    # Determine verdict
    if final_score > 0.7:
        verdict = "Likely Deepfake"
    elif final_score > 0.4:
        verdict = "Suspicious"
    else:
        verdict = "Likely Authentic"

    verdict_override = analysis_results.get('verdict')
    if isinstance(verdict_override, str) and verdict_override.strip():
        verdict = verdict_override

    findings_payload = (
        analysis_results.get('report_findings')
        or analysis_results.get('findings')
        or analysis_results.get('detailed_findings')
    )
    if isinstance(findings_payload, str):
        findings_payload = {'explanation': findings_payload}
    elif isinstance(findings_payload, list):
        findings_payload = {'findings': findings_payload}
    elif findings_payload and not isinstance(findings_payload, dict):
        # Unknown structure, convert to string for safety
        findings_payload = {'explanation': str(findings_payload)}
    if not findings_payload:
        findings_payload = {'explanation': f"The {media_type} analysis resulted in a score of {final_score:.2f}."}

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
        findings=findings_payload,
        video_metadata=video_metadata,
        audio_metadata=audio_metadata,
        show_video_score=show_video_score,
        show_audio_score=show_audio_score,
    )

    return report.output()

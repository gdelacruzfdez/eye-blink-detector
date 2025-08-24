import pandas as pd
from typing import List
from frame_info import FrameInfo
import os
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook


class BlinkDataExporter:
    def __init__(self, session_save_dir: str, frame_rate: float):
        self.session_save_dir = session_save_dir
        self.frame_rate = frame_rate

    def export_all_blink_data_to_excel(self, processed_frames: List[FrameInfo]) -> None:
        excel_file_path = f'{self.session_save_dir}/blink_data.xlsx'
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            types = ['frame', 'left', 'right']
            # Create and write summary sheets first
            for eye_type in types:
                sequences = self._identify_blink_sequences(processed_frames, eye_type)
                summary_stats = self._generate_summary_statistics(sequences, processed_frames, eye_type)
                summary_df = pd.DataFrame([summary_stats])
                summary_df.to_excel(writer, sheet_name=f'{eye_type.capitalize()} Blinks Summary', index=False)

            # Then create detailed sequence data sheets
            for eye_type in types:
                sequence_data = self._generate_sequence_data(processed_frames, eye_type)
                sequence_data.to_excel(writer, sheet_name=f'{eye_type.capitalize()} Blink Data', index=False)

            # Export frame-level data as the last sheet
            frame_data = self._generate_frame_data(processed_frames)
            frame_data.to_excel(writer, sheet_name='Frame Predictions', index=False)
        self._adjust_column_widths(excel_file_path)

    def generate_report_from_csv(self, csv_file_path: str) -> None:
        # Convert CSV data to FrameInfo objects
        processed_frames = self.read_csv_and_convert_to_frameinfo(csv_file_path)

        # Export blink data to Excel
        self.export_all_blink_data_to_excel(processed_frames)

    @staticmethod
    def read_csv_and_convert_to_frameinfo(csv_file_path: str) -> List[FrameInfo]:
        # Read the CSV file
        df = pd.read_csv(csv_file_path, sep=";")

        # Convert DataFrame to list of FrameInfo objects
        frame_info_list = []
        for index, row in df.iterrows():
            frame_info = FrameInfo(
                frame_num=row['Frame Number'],
                left_eye_pred=row['Left Eye Blink Prediction'],
                left_eye_blink_prob=row['Left Eye Blink Probability'],
                left_eye_closed_prob=row['Left Eye Closed Probability'],
                right_eye_pred=row['Right Eye Blink Prediction'],
                right_eye_blink_prob=row['Right Eye Blink Probability'],
                right_eye_closed_prob=row['Right Eye Closed Probability'],
                frame_img=None,
                frame_with_boxes=None,
                eye_boxes=None,
                left_eye_img=None,
                right_eye_img=None
            )

            frame_info_list.append(frame_info)

        return frame_info_list

    @staticmethod
    def _adjust_column_widths(excel_file_path):
        workbook = load_workbook(excel_file_path)

        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            for column in sheet.columns:
                max_length = 0
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2)  # Adding a little extra space
                sheet.column_dimensions[get_column_letter(column[0].column)].width = adjusted_width

        workbook.save(excel_file_path)

    @staticmethod
    def _generate_frame_data(processed_frames: List[FrameInfo]) -> pd.DataFrame:
        """Generates a DataFrame for frame-level data including paths, predictions, probabilities, and more."""
        data = [{
            'Frame Number': frame.frame_num,
            'Left Eye Path': os.path.join('left_eyes', f"left_eye_{frame.frame_num}.jpg"),
            'Right Eye Path': os.path.join('right_eyes', f"right_eye_{frame.frame_num}.jpg"),
            'Left Eye Blink Prediction': frame.left_eye_pred,
            'Left Eye Blink Probability': frame.left_eye_blink_prob,
            'Left Eye Closed Probability': frame.left_eye_closed_prob,
            'Right Eye Blink Prediction': frame.right_eye_pred,
            'Right Eye Blink Probability': frame.right_eye_blink_prob,
            'Right Eye Closed Probability': frame.right_eye_closed_prob,
            'Average Blink Probability': (frame.left_eye_blink_prob + frame.right_eye_blink_prob) / 2,
            'Frame Blink Prediction': 1 if ((frame.left_eye_blink_prob + frame.right_eye_blink_prob) / 2) > 0.5 else 0,
        } for frame in processed_frames]
        return pd.DataFrame(data)

    def _identify_blink_sequences(self, processed_frames: List[FrameInfo], eye_type: str) -> List[List[FrameInfo]]:
        """Identifies sequences of blinks."""
        sequences = []
        current_sequence = []

        for frame in processed_frames:
            blink_prob, _ = self._get_blink_probabilities(frame, eye_type)
            if blink_prob > 0.5:
                current_sequence.append(frame)
            elif current_sequence:
                sequences.append(current_sequence)
                current_sequence = []

        if current_sequence:  # Add any remaining sequence
            sequences.append(current_sequence)

        return sequences

    def _generate_sequence_data(self, processed_frames: List[FrameInfo], eye_type: str) -> pd.DataFrame:
        """Generates sequence-level data and appends a summary of blink statistics."""
        sequences = self._identify_blink_sequences(processed_frames, eye_type)
        data, intervals = [], []
        last_end_frame = None

        for seq_id, sequence in enumerate(sequences, start=1):
            summary, interval_frames = self._summarize_sequence(seq_id, sequence, eye_type, last_end_frame)
            data.append(summary)
            if interval_frames is not None:
                intervals.append(interval_frames)
            last_end_frame = summary['End Frame']

        return pd.DataFrame(data)

    def _generate_summary_statistics(self, sequences: List[List[FrameInfo]], processed_frames: List[FrameInfo],
                                     eye_type: str) -> dict:
        """Generate summary statistics for blink sequences."""
        total_duration = sum((seq[-1].frame_num - seq[0].frame_num + 1) / self.frame_rate for seq in sequences)
        avg_duration = total_duration / len(sequences) if sequences else 0
        num_blinks = len(sequences)

        # Separate sequences into complete and incomplete blinks
        complete_blinks_sequences = [seq for seq in sequences if any(
            frame.left_eye_closed_prob > 0.5 or frame.right_eye_closed_prob > 0.5 for frame in seq)]
        incomplete_blinks_sequences = [seq for seq in sequences if seq not in complete_blinks_sequences]
        num_complete_blinks = len(complete_blinks_sequences)
        num_incomplete_blinks = len(incomplete_blinks_sequences)

        # Calculate total and average duration for complete and incomplete blinks
        total_duration_complete = sum(
            (seq[-1].frame_num - seq[0].frame_num + 1) / self.frame_rate for seq in complete_blinks_sequences)
        avg_duration_complete = total_duration_complete / len(
            complete_blinks_sequences) if complete_blinks_sequences else 0

        total_duration_incomplete = sum(
            (seq[-1].frame_num - seq[0].frame_num + 1) / self.frame_rate for seq in incomplete_blinks_sequences)
        avg_duration_incomplete = total_duration_incomplete / len(
            incomplete_blinks_sequences) if incomplete_blinks_sequences else 0

        total_frames = len(processed_frames)
        blink_frames = sum(len(seq) for seq in sequences)
        closed_eye_frames = sum(1 for seq in sequences for frame in seq if
                                frame.left_eye_closed_prob > 0.5 or frame.right_eye_closed_prob > 0.5)
        non_blink_frames = total_frames - blink_frames

        # Calculate intervals between blinks
        intervals = [sequences[i + 1][0].frame_num - sequences[i][-1].frame_num - 1 for i in range(len(sequences) - 1)]
        avg_interval = sum(intervals) / len(intervals) / self.frame_rate if intervals else 0

        return {
            'Number of Blinks': num_blinks,
            'Number of Complete Blinks': num_complete_blinks,
            'Number of Incomplete Blinks': num_incomplete_blinks,
            'Ratio of Incomplete Blinks': num_incomplete_blinks / num_blinks if num_blinks else 0,
            'Average Blink Duration (Seconds)': avg_duration,
            'Average Complete Blink Duration (Seconds)': avg_duration_complete,
            'Average Incomplete Blink Duration (Seconds)': avg_duration_incomplete,
            'Average Interval between blinks (Seconds)': avg_interval,
            'Total Number of Frames': total_frames,
            'Number of Blink Frames': blink_frames,
            'Number of Closed Eye Frames': closed_eye_frames,
            'Number of Non-Blink Frames': non_blink_frames
        }

    def _summarize_sequence(self, seq_id: int, sequence: List[FrameInfo], eye_type: str, last_end_frame: int) -> (
    dict, int):
        """Summarizes a sequence of blinks and calculates the interval to the next blink."""
        start_frame = sequence[0].frame_num
        end_frame = sequence[-1].frame_num
        num_frames = len(sequence)
        blink_complete = any(self._get_blink_probabilities(frame, eye_type)[1] > 0.5 for frame in sequence)
        duration_seconds = num_frames / self.frame_rate
        interval_frames = start_frame - last_end_frame if last_end_frame is not None else None

        summary = {
            'Sequence ID': seq_id,
            'Start Frame': start_frame,
            'End Frame': end_frame,
            'Number of Frames': num_frames,
            'Blink Type': 'Complete' if blink_complete else 'Incomplete',
            'Duration (Seconds)': round(duration_seconds, 4),
            'Interval Since Last Blink (Frames)': interval_frames if interval_frames is not None else 'N/A',
            'Interval Since Last Blink (Seconds)': round(interval_frames / self.frame_rate,
                                                         4) if interval_frames is not None else 'N/A'
        }

        return summary, interval_frames

    @staticmethod
    def _get_blink_probabilities(frame: FrameInfo, eye_type: str) -> tuple:
        """Retrieve blink and closed probabilities based on eye type."""
        if eye_type == 'frame':
            blink_prob = (frame.left_eye_blink_prob + frame.right_eye_blink_prob) / 2
            closed_prob = max(frame.left_eye_closed_prob, frame.right_eye_closed_prob)
        else:
            blink_prob = getattr(frame, f"{eye_type}_eye_blink_prob")
            closed_prob = getattr(frame, f"{eye_type}_eye_closed_prob")
        return blink_prob, closed_prob

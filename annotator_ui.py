import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List
import glob

import cv2
import pandas as pd
from PIL import Image, ImageTk

from eye_extractor import DlibEyeExtractor


class AnnotationUI:
    def __init__(self, path: str, eyes: List[str]):
        self.eyes = [eye.upper() for eye in eyes]
        self.eye_extractor = DlibEyeExtractor()

        if os.path.isdir(path):
            self.video_list = (
                glob.glob(os.path.join(path, "*.mp4"))
                + glob.glob(os.path.join(path, "*.avi"))
                + glob.glob(os.path.join(path, "*.mov"))
            )
            self.video_list.sort()
            if not self.video_list:
                messagebox.showerror("Error", f"No video files found in {path}")
                return
            self.video_path = self.video_list[0]
        else:
            self.video_list = [path]
            self.video_path = path

        self.root = tk.Tk()
        self.root.title("Video Annotation Tool")
        self.root.geometry("1400x800")

        self.create_widgets()
        self.load_video(self.video_path)

    def load_video(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video file: {video_path}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_num = 0
        self.annotations = {}

        video_dir = os.path.dirname(self.video_path)
        video_filename = os.path.basename(self.video_path)
        self.default_save_path = os.path.join(
            video_dir, f"{video_filename}.annotations.xlsx"
        )

        if os.path.exists(self.default_save_path):
            self.load_annotations(self.default_save_path)
        else:
            self.initialize_default_annotations()

        self.tree.delete(*self.tree.get_children())
        self.populate_annotations_table()
        self.display_frame()

    def initialize_default_annotations(self):
        for i in range(self.total_frames):
            self.annotations[i] = {}
            for eye in self.eyes:
                self.annotations[i][eye] = {"blink": 0, "NV": 0, "blink_id": -1}

    def load_annotations(self, path: str):
        df = pd.read_excel(path)
        self.annotations = {}
        for index, row in df.iterrows():
            frame_num = row["frameId"]
            eye = row["eye"]
            if frame_num not in self.annotations:
                self.annotations[frame_num] = {}
            self.annotations[frame_num][eye] = {
                "blink": row["blink"],
                "NV": row["NV"],
                "blink_id": row["blink_id"],
            }

    def create_widgets(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=1)

        video_list_panel = tk.Frame(main_pane)
        self.create_video_list_panel(video_list_panel)
        main_pane.add(video_list_panel, width=200)

        video_panel = tk.Frame(main_pane)
        self.create_video_panel(video_panel)
        main_pane.add(video_panel, width=800)

        side_panel = tk.Frame(main_pane)
        self.create_side_panel(side_panel)
        main_pane.add(side_panel)

        # Menu
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save", command=self.save_annotations)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        # Key bindings
        self.root.bind("<Up>", lambda event: self.prev_frame())
        self.root.bind("<Down>", lambda event: self.next_frame())
        self.root.bind("0", lambda event: self.annotate_both_eyes(0))
        self.root.bind("1", lambda event: self.annotate_both_eyes(1))
        self.root.bind("2", lambda event: self.annotate_both_eyes(2))

    def create_video_list_panel(self, parent):
        label = tk.Label(parent, text="Videos", font=("Arial", 12, "bold"))
        label.pack()

        self.video_listbox = tk.Listbox(parent)
        for video in self.video_list:
            self.video_listbox.insert(tk.END, os.path.basename(video))
        self.video_listbox.pack(fill=tk.BOTH, expand=1)
        self.video_listbox.bind("<<ListboxSelect>>", self.on_video_select)

    def on_video_select(self, event):
        selection_index = self.video_listbox.curselection()
        if selection_index:
            video_index = selection_index[0]
            self.load_video(self.video_list[video_index])

    def create_video_panel(self, parent):
        # Frame display
        self.frame_label = tk.Label(parent)
        self.frame_label.pack()

        # Eye displays
        eye_frame = tk.Frame(parent)
        self.eye_labels = {}
        for eye in self.eyes:
            self.eye_labels[eye] = tk.Label(eye_frame)
            self.eye_labels[eye].pack(side=tk.LEFT, padx=10)
        eye_frame.pack()

        # Annotation buttons
        annotation_frame = tk.Frame(parent)
        for eye in self.eyes:
            button_frame = self.create_annotation_buttons(f"{eye.capitalize()} Eye", annotation_frame)
            button_frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.X)
        annotation_frame.pack(fill=tk.X, expand=True)

        # Navigation buttons
        nav_frame = tk.Frame(parent)
        self.prev_button = tk.Button(nav_frame, text="<< Previous", command=self.prev_frame)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button = tk.Button(nav_frame, text="Next >>", command=self.next_frame)
        self.next_button.pack(side=tk.RIGHT)
        self.frame_counter_label = tk.Label(nav_frame, text="Frame: 0/0")
        self.frame_counter_label.pack(side=tk.LEFT, expand=True)
        nav_frame.pack()

    def create_annotation_buttons(self, title: str, parent) -> tk.Frame:
        frame = tk.Frame(parent, borderwidth=2, relief=tk.GROOVE)
        label = tk.Label(frame, text=title, font=("Arial", 12, "bold"))
        label.pack()

        no_blink_btn = tk.Button(
            frame, text="No Blink (0)", command=lambda: self.annotate(title.split()[0].lower(), 0)
        )
        no_blink_btn.pack()

        partial_btn = tk.Button(
            frame,
            text="Partially Closed (1)",
            command=lambda: self.annotate(title.split()[0].lower(), 1),
        )
        partial_btn.pack()

        full_btn = tk.Button(
            frame,
            text="Fully Closed (2)",
            command=lambda: self.annotate(title.split()[0].lower(), 2),
        )
        full_btn.pack()

        nv_btn = tk.Button(
            frame, text="Not Visible", command=lambda: self.annotate(title.split()[0].lower(), -1)
        )
        nv_btn.pack()
        return frame

    def create_side_panel(self, parent):
        label = tk.Label(parent, text="Annotations", font=("Arial", 12, "bold"))
        label.pack()

        columns = ["Frame"]
        for eye in self.eyes:
            columns.extend([f"{eye.capitalize()} Blink", f"{eye.capitalize()} NV", f"{eye.capitalize()} Blink ID"])

        self.tree = ttk.Treeview(parent, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80, stretch=tk.NO)

        self.tree.pack(fill=tk.BOTH, expand=1)
        self.tree.bind("<<TreeviewSelect>>", self.on_table_row_click)

    def populate_annotations_table(self, start_index=0, chunk_size=100):
        end_index = min(start_index + chunk_size, self.total_frames)
        for i in range(start_index, end_index):
            values = [i]
            for eye in self.eyes:
                annotation = self.annotations[i][eye]
                blink_text = ["No Blink", "Partially Closed", "Fully Closed"][annotation["blink"]]
                values.extend([blink_text, annotation["NV"], annotation["blink_id"]])
            self.tree.insert("", "end", values=tuple(values))

        if end_index < self.total_frames:
            self.root.after(
                10, self.populate_annotations_table, end_index, chunk_size
            )

    def update_full_table_annotations(self):
        self.generate_blink_ids()
        for i, item in enumerate(self.tree.get_children()):
            values = [i]
            for eye in self.eyes:
                annotation = self.annotations[i][eye]
                blink_text = ["No Blink", "Partially Closed", "Fully Closed"][annotation["blink"]]
                values.extend([blink_text, annotation["NV"], annotation["blink_id"]])
            self.tree.item(item, values=tuple(values))

    def on_table_row_click(self, event):
        for selected_item in self.tree.selection():
            item = self.tree.item(selected_item)
            frame_num = int(item["values"][0])
            if frame_num != self.current_frame_num:
                self.current_frame_num = frame_num
                self.display_frame()

    def display_frame(self):
        logging.info(f"Displaying frame {self.current_frame_num}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        ret, frame = self.cap.read()
        if not ret:
            logging.warning(f"Could not read frame {self.current_frame_num}")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img.resize((640, 480)))

        self.frame_label.config(image=img_tk)
        self.frame_label.image = img_tk

        self.update_frame_counter()
        self.extract_and_display_eyes(frame)
        self.update_table_selection()

    def update_frame_counter(self):
        self.frame_counter_label.config(
            text=f"Frame: {self.current_frame_num}/{self.total_frames}"
        )

    def update_table_selection(self):
        for item in self.tree.get_children():
            if int(self.tree.item(item)["values"][0]) == self.current_frame_num:
                self.tree.selection_set(item)
                self.tree.see(item)
                break

    def extract_and_display_eyes(self, frame):
        left_eyes, right_eyes, _, _ = self.eye_extractor.extract(frame)
        eye_images = {"LEFT": left_eyes, "RIGHT": right_eyes}

        for eye_str, eye_imgs in eye_images.items():
            if eye_str in self.eye_labels:
                if eye_imgs:
                    eye_img = eye_imgs[0].resize((100, 50))
                    eye_tk = ImageTk.PhotoImage(image=eye_img)
                    self.eye_labels[eye_str].config(image=eye_tk)
                    self.eye_labels[eye_str].image = eye_tk
                else:
                    self.eye_labels[eye_str].config(image=None)
                    self.eye_labels[eye_str].image = None

    def next_frame(self):
        if self.current_frame_num < self.total_frames - 1:
            self.current_frame_num += 1
            self.display_frame()

    def prev_frame(self):
        if self.current_frame_num > 0:
            self.current_frame_num -= 1
            self.display_frame()

    def annotate(self, eye: str, blink_status: int):
        eye_upper = eye.upper()
        current_annotation = self.annotations[self.current_frame_num][eye_upper]

        if blink_status == -1:  # Not Visible button clicked
            nv_status = 1 - current_annotation["NV"]  # Toggle NV status
            blink_status = current_annotation["blink"]
        else:
            nv_status = 0  # If any other button is clicked, NV is 0
            current_annotation["blink"] = blink_status

        current_annotation["NV"] = nv_status

        self.update_full_table_annotations()
        self.save_annotations()

    def annotate_both_eyes(self, blink_status: int):
        for eye in self.eyes:
            self.annotate(eye.lower(), blink_status)

    def generate_blink_ids(self):
        for eye in self.eyes:
            in_blink = False
            blink_id_counter = 1
            for frame_num in range(self.total_frames):
                annotation = self.annotations[frame_num][eye]
                is_blinking = annotation["blink"] in [1, 2]

                if is_blinking and not in_blink:
                    in_blink = True
                    annotation["blink_id"] = blink_id_counter
                elif is_blinking and in_blink:
                    annotation["blink_id"] = blink_id_counter
                elif not is_blinking and in_blink:
                    in_blink = False
                    blink_id_counter += 1
                    annotation["blink_id"] = -1
                else:  # not is_blinking and not in_blink
                    annotation["blink_id"] = -1

    def save_annotations(self):
        if not self.annotations:
            messagebox.showwarning("No annotations", "There are no annotations to save.")
            return

        self.generate_blink_ids()

        records = []
        for frame_num, frame_annotations in self.annotations.items():
            for eye, annotation in frame_annotations.items():
                records.append(
                    {
                        "video": self.video_path,
                        "frameId": frame_num,
                        "eye": eye,
                        "blink": annotation["blink"],
                        "NV": annotation["NV"],
                        "blink_id": annotation["blink_id"],
                    }
                )

        df = pd.DataFrame(records)
        df.to_excel(self.default_save_path, index=False)
        logging.info(f"Annotations saved to {self.default_save_path}")

    def run(self):
        self.root.mainloop()
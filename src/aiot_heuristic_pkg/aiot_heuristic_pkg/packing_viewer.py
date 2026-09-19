"""
packing_current.png가 바뀔 때마다 별도 GUI 창에 표시
"""

import sys
import tkinter as tk
from math import ceil
from pathlib import Path


REFRESH_MS = 200


class PackingViewer:
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.last_modified = None
        self.photo = None
        self.running = True

        self.root = tk.Tk()
        self.root.title("AIOT packing live view")
        self.root.configure(background="black")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.label = tk.Label(self.root, text="Waiting for first packing plan", foreground="white", background="black")
        self.label.pack(fill="both", expand=True)

        self.root.after(0, self.refresh)

    def refresh(self):
        if not self.running:
            return

        try:
            modified = self.image_path.stat().st_mtime_ns

            if modified != self.last_modified:
                photo = tk.PhotoImage(file=str(self.image_path))
                screen_width = max(1, self.root.winfo_screenwidth() - 80)
                screen_height = max(1, self.root.winfo_screenheight() - 120)
                scale = max(1, ceil(photo.width() / screen_width), ceil(photo.height() / screen_height))

                if scale > 1:
                    photo = photo.subsample(scale, scale)

                self.photo = photo
                self.label.configure(image=self.photo, text="")
                self.last_modified = modified
                self.root.geometry(f"{photo.width()}x{photo.height()}")
        except (FileNotFoundError, OSError, tk.TclError):
            # 첫 frame 생성 전이거나 파일 교체 중이면 다음 주기에 다시 읽음
            pass

        self.root.after(REFRESH_MS, self.refresh)

    def close(self):
        self.running = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(args=None):
    values = sys.argv[1:] if args is None else args

    if len(values) != 1:
        raise SystemExit("usage: packing_viewer <packing_current.png>")

    viewer = PackingViewer(values[0])
    viewer.run()


if __name__ == "__main__":
    main()

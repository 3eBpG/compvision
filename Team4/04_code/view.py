import tkinter as tk
from tkinter import filedialog
import open3d as o3d

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Выбери .ply файл",
    filetypes=[("Point Cloud", "*.ply"), ("All files", "*.*")],
)

if file_path:
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd], window_name=file_path)
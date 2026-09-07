import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading

# 导入核心逻辑
from logic_models import ModelManager
from logic_roi import apply_roi_filter
from logic_tape import run_tape_segmentation
from logic_branch import run_branch_segmentation
from logic_trunk import run_trunk_extraction
from logic_bud import run_bud_global_detection, run_bud_local_detection
from logic_skeleton import (
    run_direct_skeletonization,
    run_skeleton_topology,
    draw_direct_skeleton_visualization,
    draw_topology_visualization,
)

class UnifiedSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("樱桃树综合分析系统 (综合整合版)")
        self.root.geometry("1500x900")
        
        # 状态变量
        self.img_rgb = None
        self.img_bgr = None
        self.photos = {}
        
        self.var_tape = tk.BooleanVar(value=False)
        self.var_branch = tk.BooleanVar(value=False)
        self.var_trunk = tk.BooleanVar(value=False)
        self.var_bud_global = tk.BooleanVar(value=False)
        self.var_bud_local = tk.BooleanVar(value=False)
        self.var_direct_skeleton = tk.BooleanVar(value=False)
        self.var_skeleton = tk.BooleanVar(value=False)
        self.var_roi_boundary = tk.BooleanVar(value=False)
        self.var_structure_black = tk.BooleanVar(value=False)
        
        # 视图缩放与平移参数
        self.view_scale = 1.0
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # 局部芽点框参数
        self.box_size = 256
        self.box_x = 0
        self.box_y = 0
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.scale_ratio = 1.0
        
        # 缓存数据
        self.cached_roi_mask = None
        self.cached_roi_filtered_rgb = None
        self.cached_branch_mask = None
        self.cached_trunk_mask = None
        self.cached_refined_branch_mask = None
        self.cached_direct_skeleton = None
        self.cached_global_boxes = []
        self.cached_global_scores = []
        self.cached_global_labels = []
        self.cached_global_masks = []
        
        self.init_gui()
        
    def init_gui(self):
        # 顶部控制面板
        top_frame = tk.Frame(self.root, bg='#f0f0f0')
        top_frame.pack(fill=tk.X, side=tk.TOP, pady=5)
        
        # 按钮区
        btn_frame = tk.Frame(top_frame, bg='#f0f0f0')
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(btn_frame, text="打开图片", command=self.open_image, bg='#4CAF50', fg='white', font=('Arial', 12), padx=20).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="开始综合分析", command=self.start_processing, bg='#2196F3', fg='white', font=('Arial', 12), padx=20).pack(side=tk.LEFT, padx=10)
        
        self.lbl_status = tk.Label(btn_frame, text="就绪", bg='#f0f0f0', font=('Arial', 10))
        self.lbl_status.pack(side=tk.LEFT, padx=20)
        
        # 功能选择区
        opt_frame = tk.Frame(top_frame, bg='#f0f0f0')
        opt_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(opt_frame, text="选择叠加功能:", bg='#f0f0f0', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=10)
        
        cb_tape = tk.Checkbutton(opt_frame, text="1. 条带分割", variable=self.var_tape, bg='#f0f0f0')
        cb_tape.pack(side=tk.LEFT, padx=10)
        
        cb_branch = tk.Checkbutton(opt_frame, text="2. 树枝分割", variable=self.var_branch, bg='#f0f0f0')
        cb_branch.pack(side=tk.LEFT, padx=10)
        
        cb_trunk = tk.Checkbutton(opt_frame, text="3. 主干提取 (可配合条带使用)", variable=self.var_trunk, bg='#f0f0f0')
        cb_trunk.pack(side=tk.LEFT, padx=10)
        
        cb_bud_g = tk.Checkbutton(opt_frame, text="4. 全图芽点检测", variable=self.var_bud_global, bg='#f0f0f0', command=self.on_bud_g_toggle)
        cb_bud_g.pack(side=tk.LEFT, padx=10)
        
        cb_bud_l = tk.Checkbutton(opt_frame, text="5. 局部芽点检测 (鼠标框选)", variable=self.var_bud_local, bg='#f0f0f0', command=self.on_bud_l_toggle)
        cb_bud_l.pack(side=tk.LEFT, padx=10)
        
        cb_direct_skel = tk.Checkbutton(opt_frame, text="6. 直接细化骨架 (依赖2)", variable=self.var_direct_skeleton, bg='#f0f0f0')
        cb_direct_skel.pack(side=tk.LEFT, padx=10)

        cb_skel = tk.Checkbutton(opt_frame, text="7. 2D拓扑图构建 (依赖2/3/4)", variable=self.var_skeleton, bg='#f0f0f0')
        cb_skel.pack(side=tk.LEFT, padx=10)
        
        cb_roi = tk.Checkbutton(opt_frame, text="8. 显示ROI边界", variable=self.var_roi_boundary, bg='#f0f0f0')
        cb_roi.pack(side=tk.LEFT, padx=10)

        cb_structure = tk.Checkbutton(opt_frame, text="9. 黑底结构视图", variable=self.var_structure_black, bg='#f0f0f0')
        cb_structure.pack(side=tk.LEFT, padx=10)
        
        # 主内容区域
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # 左侧原图
        left_frame = tk.Frame(main_frame, bg='#e0e0e0')
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        tk.Label(left_frame, text="原图与交互区域", bg='#e0e0e0', font=('Arial', 12, 'bold')).pack(pady=5)
        self.canvas_orig = tk.Canvas(left_frame, bg='white', cursor='crosshair')
        self.canvas_orig.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 绑定鼠标事件用于局部芽点检测
        self.canvas_orig.bind('<Button-1>', self.on_mouse_down)
        self.canvas_orig.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas_orig.bind('<ButtonRelease-1>', self.on_mouse_up)
        
        # 右侧结果
        right_frame = tk.Frame(main_frame, bg='#e0e0e0')
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        tk.Label(right_frame, text="综合叠加结果", bg='#e0e0e0', font=('Arial', 12, 'bold')).pack(pady=5)
        self.canvas_result = tk.Canvas(right_frame, bg='white')
        self.canvas_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 绑定缩放与平移 (左右画布联动)
        for canvas in [self.canvas_orig, self.canvas_result]:
            canvas.bind('<MouseWheel>', self.on_mouse_wheel)  # Windows
            canvas.bind('<Button-4>', self.on_mouse_wheel)    # Linux
            canvas.bind('<Button-5>', self.on_mouse_wheel)    # Linux
            canvas.bind('<Button-3>', self.on_pan_start)      # 鼠标右键平移
            canvas.bind('<B3-Motion>', self.on_pan_drag)
            canvas.bind('<ButtonRelease-3>', self.on_pan_end)

    def on_bud_g_toggle(self):
        if self.var_bud_global.get():
            self.var_bud_local.set(False)
            self.draw_orig()

    def on_bud_l_toggle(self):
        if self.var_bud_local.get():
            self.var_bud_global.set(False)
            self.draw_orig()
        else:
            self.draw_orig()

    def open_image(self):
        filepath = filedialog.askopenfilename(title="选择图片", filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")])
        if filepath:
            self.img_bgr = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.img_bgr is None:
                messagebox.showerror("错误", "无法读取图片")
                return
            self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
            
            # 清除缓存
            self.cached_roi_mask = None
            self.cached_roi_filtered_rgb = None
            self.cached_branch_mask = None
            self.cached_trunk_mask = None
            self.cached_refined_branch_mask = None
            self.cached_direct_skeleton = None
            self.cached_global_boxes = []
            self.cached_global_scores = []
            self.cached_global_labels = []
            self.cached_global_masks = []
            
            # 初始化视图参数
            self.view_scale = 1.0
            self.view_offset_x = 0
            self.view_offset_y = 0
            
            # 初始化局部框位置在中心
            h, w = self.img_rgb.shape[:2]
            self.box_x = max(0, (w - self.box_size) // 2)
            self.box_y = max(0, (h - self.box_size) // 2)
            
            self.canvas_result.delete("all")
            self.draw_orig()
            self.lbl_status.config(text=f"已加载图片: {os.path.basename(filepath)}")

    def draw_orig(self):
        if self.img_rgb is None: return
        
        cw = self.canvas_orig.winfo_width()
        ch = self.canvas_orig.winfo_height()
        if cw <= 1 or ch <= 1:
            self.root.after(100, self.draw_orig)
            return
            
        h, w = self.img_rgb.shape[:2]
        base_scale = min(cw / w, ch / h) * 0.95
        self.scale_ratio = base_scale * self.view_scale
        
        nw = max(1, int(w * self.scale_ratio))
        nh = max(1, int(h * self.scale_ratio))
        resized = cv2.resize(self.img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        
        pil_img = Image.fromarray(resized)
        self.photos['orig'] = ImageTk.PhotoImage(pil_img)
        
        self.canvas_orig.delete("all")
        
        # 应用平移偏移量
        center_x = cw//2 + self.view_offset_x
        center_y = ch//2 + self.view_offset_y
        self.canvas_orig.create_image(center_x, center_y, image=self.photos['orig'])
        
        # 绘制局部检测框
        if self.var_bud_local.get():
            bx = center_x - nw//2 + int(self.box_x * self.scale_ratio)
            by = center_y - nh//2 + int(self.box_y * self.scale_ratio)
            bw = int(self.box_size * self.scale_ratio)
            bh = int(self.box_size * self.scale_ratio)
            
            self.canvas_orig.create_rectangle(bx, by, bx+bw, by+bh, outline='red', width=3, tags="box")
            self.canvas_orig.create_text(bx+5, by+5, text="拖动此框", fill='red', anchor='nw', tags="box")

    def on_mouse_down(self, event):
        if not self.var_bud_local.get() or self.img_rgb is None: return
        
        cw = self.canvas_orig.winfo_width()
        ch = self.canvas_orig.winfo_height()
        h, w = self.img_rgb.shape[:2]
        nw = int(w * self.scale_ratio)
        nh = int(h * self.scale_ratio)
        
        center_x = cw//2 + self.view_offset_x
        center_y = ch//2 + self.view_offset_y
        
        bx = center_x - nw//2 + int(self.box_x * self.scale_ratio)
        by = center_y - nh//2 + int(self.box_y * self.scale_ratio)
        bw = int(self.box_size * self.scale_ratio)
        bh = int(self.box_size * self.scale_ratio)
        
        if bx <= event.x <= bx+bw and by <= event.y <= by+bh:
            self.is_dragging = True
            self.drag_start_x = event.x
            self.drag_start_y = event.y

    def on_mouse_drag(self, event):
        if self.is_dragging:
            dx = (event.x - self.drag_start_x) / self.scale_ratio
            dy = (event.y - self.drag_start_y) / self.scale_ratio
            
            self.box_x = max(0, min(self.img_rgb.shape[1] - self.box_size, int(self.box_x + dx)))
            self.box_y = max(0, min(self.img_rgb.shape[0] - self.box_size, int(self.box_y + dy)))
            
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.draw_orig()

    def on_mouse_up(self, event):
        self.is_dragging = False

    def update_status(self, text):
        self.lbl_status.config(text=text)
        self.root.update()

    def on_mouse_wheel(self, event):
        if self.img_rgb is None: return
        
        # 判断滚动方向
        if event.num == 5 or event.delta < 0:
            scale_factor = 0.9  # 缩小
        else:
            scale_factor = 1.1  # 放大
            
        # 限制缩放范围
        new_scale = self.view_scale * scale_factor
        if 0.1 <= new_scale <= 10.0:
            self.view_scale = new_scale
            # 缩放时保持当前偏移比例
            self.view_offset_x = int(self.view_offset_x * scale_factor)
            self.view_offset_y = int(self.view_offset_y * scale_factor)
            
            self.draw_orig()
            # 如果右侧有图，也重新绘制
            if hasattr(self, 'final_result') and self.final_result is not None:
                self.show_result(self.final_result)

    def on_pan_start(self, event):
        if self.img_rgb is None: return
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def on_pan_drag(self, event):
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            self.view_offset_x += dx
            self.view_offset_y += dy
            
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            
            self.draw_orig()
            if hasattr(self, 'final_result') and self.final_result is not None:
                self.show_result(self.final_result)

    def on_pan_end(self, event):
        self.is_panning = False

    def show_result(self, result_rgb):
        self.final_result = result_rgb  # 缓存一份用于缩放平移重绘
        cw = self.canvas_result.winfo_width()
        ch = self.canvas_result.winfo_height()
        if cw <= 1 or ch <= 1:
            self.root.after(100, lambda: self.show_result(result_rgb))
            return
            
        h, w = result_rgb.shape[:2]
        base_scale = min(cw / w, ch / h) * 0.95
        ratio = base_scale * self.view_scale
        
        nw = max(1, int(w * ratio))
        nh = max(1, int(h * ratio))
        resized = cv2.resize(result_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        
        pil_img = Image.fromarray(resized)
        self.photos['result'] = ImageTk.PhotoImage(pil_img)
        
        self.canvas_result.delete("all")
        
        # 应用平移偏移量
        center_x = cw//2 + self.view_offset_x
        center_y = ch//2 + self.view_offset_y
        self.canvas_result.create_image(center_x, center_y, image=self.photos['result'])

    def draw_buds(self, overlay, boxes, scores, labels, masks_info, patch_size=256):
        color_map = {0: (255, 0, 0), 1: (0, 255, 0)} # 花芽红，叶芽绿
        if boxes is not None and len(boxes) > 0:
            for box, score, label, mask_data in zip(boxes, scores, labels, masks_info):
                x1, y1, x2, y2 = map(int, box)
                color = color_map.get(int(label), (0, 255, 255))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                local_mask, offset_x, offset_y = mask_data
                if local_mask.shape[0] == patch_size and local_mask.shape[1] == patch_size:
                    # 对于全局，mask_data包含完整的patch大小
                    h_m, w_m = local_mask.shape
                    # 确保不越界
                    h_img, w_img = overlay.shape[:2]
                    end_y = min(offset_y + h_m, h_img)
                    end_x = min(offset_x + w_m, w_img)
                    valid_h = end_y - offset_y
                    valid_w = end_x - offset_x
                    
                    global_bool_mask = np.zeros(overlay.shape[:2], dtype=bool)
                    global_bool_mask[offset_y:end_y, offset_x:end_x] = local_mask[:valid_h, :valid_w]
                    overlay[global_bool_mask] = color

    def start_processing(self):
        if self.img_rgb is None:
            messagebox.showwarning("警告", "请先打开图片")
            return
            
        def process_thread():
            try:
                mm = ModelManager()
                
                req_roi = (
                    self.var_branch.get()
                    or self.var_trunk.get()
                    or self.var_bud_global.get()
                    or self.var_direct_skeleton.get()
                    or self.var_skeleton.get()
                    or self.var_roi_boundary.get()
                )
                
                roi_mask = None
                roi_filtered_rgb = self.img_rgb.copy()
                
                # 步骤 1: ROI 过滤
                if req_roi:
                    if self.cached_roi_mask is not None and self.cached_roi_filtered_rgb is not None:
                        self.update_status("使用缓存的 ROI 结果...")
                        roi_mask = self.cached_roi_mask
                        roi_filtered_rgb = self.cached_roi_filtered_rgb
                    else:
                        self.update_status("正在加载 ROI 模型...")
                        roi_model = mm.load_roi_model()
                        self.update_status("正在提取 ROI...")
                        roi_mask, roi_filtered_rgb = apply_roi_filter(roi_model, self.img_rgb)
                        self.cached_roi_mask = roi_mask
                        self.cached_roi_filtered_rgb = roi_filtered_rgb
                
                # 准备叠加图底板
                structure_black_mode = self.var_structure_black.get()
                if structure_black_mode:
                    overlay = np.zeros_like(self.img_rgb)
                else:
                    overlay = self.img_rgb.copy()
                
                # 步骤 2: 条带分割
                tape_mask = None
                if self.var_tape.get():
                    self.update_status("正在加载条带分割模型...")
                    tape_model = mm.load_tape_model()
                    self.update_status("正在分割标定带...")
                    tape_mask = run_tape_segmentation(tape_model, self.img_rgb)
                    if structure_black_mode:
                        overlay[tape_mask == 255] = [255, 255, 255]
                    else:
                        # 黄色叠加
                        overlay[tape_mask == 255] = [255, 255, 0]
                
                # 步骤 3 & 4: 树枝与主干
                branch_mask = None
                trunk_mask = None
                refined_branch = None
                if self.var_branch.get() or self.var_trunk.get() or self.var_direct_skeleton.get() or self.var_skeleton.get():
                    if self.cached_branch_mask is not None:
                        self.update_status("使用缓存的枝条分割结果...")
                        branch_mask = self.cached_branch_mask
                    else:
                        self.update_status("正在加载枝条分割模型...")
                        branch_model = mm.load_branch_model()
                        self.update_status("正在分割樱桃树枝...")
                        branch_mask = run_branch_segmentation(branch_model, roi_filtered_rgb)
                        self.cached_branch_mask = branch_mask
                    
                    need_trunk = self.var_trunk.get() or self.var_skeleton.get()
                    if need_trunk:
                        if self.cached_trunk_mask is not None and self.cached_refined_branch_mask is not None:
                            self.update_status("使用缓存的主干提取结果...")
                            trunk_mask = self.cached_trunk_mask
                            refined_branch = self.cached_refined_branch_mask
                        else:
                            self.update_status("正在计算动态主干提取...")
                            trunk_mask, refined_branch = run_trunk_extraction(branch_mask, tape_mask)
                            self.cached_trunk_mask = trunk_mask
                            self.cached_refined_branch_mask = refined_branch

                    if self.var_direct_skeleton.get():
                        if self.cached_direct_skeleton is not None:
                            self.update_status("使用缓存的直接细化骨架结果...")
                            direct_skeleton_result = self.cached_direct_skeleton
                        else:
                            self.update_status("正在生成旧版直接细化骨架...")
                            direct_skeleton_result = run_direct_skeletonization(branch_mask, prune_length=6)
                            self.cached_direct_skeleton = direct_skeleton_result
                        overlay = draw_direct_skeleton_visualization(overlay, direct_skeleton_result)

                    if self.var_trunk.get() and trunk_mask is not None and refined_branch is not None:
                        if structure_black_mode:
                            # 黑底结构视图: 主干蓝色，其余枝条白色
                            overlay[refined_branch == 255] = [255, 255, 255]
                            overlay[trunk_mask == 255] = [0, 0, 255]
                        else:
                            # 红色主干，绿色枝条
                            overlay[trunk_mask == 255] = [255, 0, 0]
                            overlay[refined_branch == 255] = [0, 255, 0]
                    elif self.var_branch.get():
                        if structure_black_mode:
                            # 黑底结构视图: 枝条白色
                            overlay[branch_mask == 255] = [255, 255, 255]
                        else:
                            # 仅枝条，用蓝色
                            overlay[branch_mask == 255] = [0, 0, 255]
                
                # 步骤 5: 全图芽点
                global_boxes, global_scores, global_labels, global_masks = [], [], [], []
                if self.var_bud_global.get() or self.var_skeleton.get():
                    if len(self.cached_global_boxes) > 0:
                        self.update_status("使用缓存的全图芽点检测结果...")
                        global_boxes = self.cached_global_boxes
                        global_scores = self.cached_global_scores
                        global_labels = self.cached_global_labels
                        global_masks = self.cached_global_masks
                    else:
                        self.update_status("正在加载全图芽点模型...")
                        bud_g_model = mm.load_bud_global_model()
                        self.update_status("正在进行全图芽点检测(这需要较长时间)...")
                        
                        roi_filtered_bgr = cv2.cvtColor(roi_filtered_rgb, cv2.COLOR_RGB2BGR)
                        
                        def prog_cb(curr, total):
                            self.update_status(f"全图芽点检测进度: {curr}/{total}")
                        
                        global_boxes, global_scores, global_labels, global_masks = run_bud_global_detection(bud_g_model, roi_filtered_bgr, progress_callback=prog_cb)
                        
                        self.cached_global_boxes = global_boxes
                        self.cached_global_scores = global_scores
                        self.cached_global_labels = global_labels
                        self.cached_global_masks = global_masks
                        
                    if self.var_bud_global.get():
                        self.draw_buds(overlay, global_boxes, global_scores, global_labels, global_masks, patch_size=256)
                
                # 步骤 6: 局部芽点
                if self.var_bud_local.get():
                    self.update_status("正在加载局部芽点模型...")
                    bud_l_model = mm.load_bud_local_model()
                    self.update_status("正在进行局部芽点检测...")
                    
                    box_coords = (self.box_x, self.box_y, self.box_size, self.box_size)
                    boxes, scores, labels, masks = run_bud_local_detection(bud_l_model, self.img_bgr, box_coords)
                    
                    # 画出局部检测框边界
                    cv2.rectangle(overlay, (self.box_x, self.box_y), (self.box_x+self.box_size, self.box_y+self.box_size), (0, 0, 255), 2)
                    
                    if len(boxes) > 0:
                        for box, score, label, mask_data in zip(boxes, scores, labels, masks):
                            x1, y1, x2, y2 = map(int, box)
                            color = (255, 0, 0) if int(label) == 0 else (0, 255, 0)
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(overlay, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
                            local_mask, offset_x, offset_y = mask_data
                            h_m, w_m = local_mask.shape
                            global_bool_mask = np.zeros(overlay.shape[:2], dtype=bool)
                            # local_mask尺寸与patch(256x256)一致
                            end_y = min(offset_y + h_m, overlay.shape[0])
                            end_x = min(offset_x + w_m, overlay.shape[1])
                            global_bool_mask[offset_y:end_y, offset_x:end_x] = local_mask[:(end_y-offset_y), :(end_x-offset_x)]
                            overlay[global_bool_mask] = color
                
                # 步骤 7: 骨架与拓扑校验
                if self.var_skeleton.get():
                    self.update_status("正在构建 2D 骨架拓扑图...")
                    if trunk_mask is not None and refined_branch is not None:
                        topology_result = run_skeleton_topology(
                            trunk_mask=trunk_mask,
                            branch_mask=refined_branch,
                            bud_boxes=global_boxes,
                            bud_masks_info=global_masks,
                            img_shape=self.img_rgb.shape,
                        )
                        overlay = draw_topology_visualization(overlay, topology_result)
                    else:
                        self.update_status("未获取到主干/枝条掩码，无法构建 2D 拓扑图")
                
                # 合成最终结果
                self.update_status("正在生成最终可视化结果...")
                if structure_black_mode:
                    final_result = overlay.copy()
                else:
                    final_result = cv2.addWeighted(self.img_rgb, 0.4, overlay, 0.6, 0)
                
                # 步骤 8: 绘制ROI边界
                if self.var_roi_boundary.get() and roi_mask is not None:
                    self.update_status("正在绘制ROI边界...")
                    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # 绘制显眼的品红色线条(BGR为Magenta/紫红)，线宽3
                    cv2.drawContours(final_result, contours, -1, (255, 0, 255), 3)
                
                self.show_result(final_result)
                self.update_status("处理完成！")
                
            except Exception as e:
                self.update_status("处理发生错误")
                messagebox.showerror("错误", f"处理过程中发生异常:\n{str(e)}")

        # 在新线程运行避免卡死GUI
        threading.Thread(target=process_thread, daemon=True).start()

def main():
    root = tk.Tk()
    app = UnifiedSystemGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()

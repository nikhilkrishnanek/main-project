
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtGui import QColor, QFont

class TargetTable(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # Title Header
        title_layout = QVBoxLayout()
        self.lbl_title = QLabel("📄 ACTIVE TARGET INVENTORY")
        self.lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ff00; margin-bottom: 5px;")
        layout.addWidget(self.lbl_title)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ID", "RANGE (m)", "THREAT", "VELOCITY (m/s)", "CLASSIFICATION", "STATUS"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        
        # Styling to match the "clean" look within dark theme context
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #f0f0f0; 
                color: #333; 
                gridline-color: #ddd;
                border: none;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                color: #888;
                padding: 8px;
                border: none;
                font-weight: bold;
                text-transform: uppercase;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #e0e0e0;
                color: #000;
            }
        """)
        
        layout.addWidget(self.table)
        
        # Export Button (Keep existing)
        self.btn_export = QPushButton("EXPORT TARGET DATA (CSV)")
        self.btn_export.setStyleSheet("background-color: #333; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.btn_export.clicked.connect(self._export_data)
        layout.addWidget(self.btn_export)
        
        self.current_tracks = []
        
    def update_table(self, frame_data):
        self.current_tracks = frame_data.get('tracks', [])
        tracks = self.current_tracks
        
        self.table.setRowCount(len(tracks))
        
        for row, track in enumerate(tracks):
            tid = track['id']
            rng = track['range_m']
            vel = track['velocity_m_s']
            cls = track.get('label', track.get('description', 'Unknown')).upper()
            if cls == "UNKNOWN": cls = track.get('class_label', 'SCANNING...').upper()

            # Logic for Threat and Status
            threat_score = 0
            status = "NOMINAL"
            status_color = "#00cc00" # Green
            
            # Simple Threat Logic
            if abs(vel) > 20: threat_score += 30
            if rng < 500: threat_score += 20
            if "MISSILE" in cls or "FIGHTER" in cls: threat_score += 40
            
            if threat_score > 60:
                status = "CRITICAL"
                status_color = "#ff0000"
            elif threat_score > 30:
                status = "ELEVATED"
                status_color = "#ffaa00"
            
            # ID Formatting
            id_str = f"INV-{tid:03d}"
            
            # Items
            item_id = QTableWidgetItem(id_str)
            item_rng = QTableWidgetItem(f"{rng:.1f}")
            item_threat = QTableWidgetItem(str(threat_score))
            item_vel = QTableWidgetItem(f"{vel:.1f}")
            item_cls = QTableWidgetItem(cls)
            item_status = QTableWidgetItem(status)
            
            # Apply Status Color
            item_status.setForeground(QColor(status_color))
            item_status.setFont(QFont("Segoe UI", 9, QFont.Bold))
            
            self.table.setItem(row, 0, item_id)
            self.table.setItem(row, 1, item_rng)
            self.table.setItem(row, 2, item_threat)
            self.table.setItem(row, 3, item_vel)
            self.table.setItem(row, 4, item_cls)
            self.table.setItem(row, 5, item_status)

    def _export_data(self):
        if not self.current_tracks:
            return
            
        import csv
        import datetime
        from PySide6.QtWidgets import QFileDialog
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, _ = QFileDialog.getSaveFileName(self, "Save Target Report", f"targets_{timestamp}.csv", "CSV Files (*.csv)")
        
        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Header
                    writer.writerow(["Track ID", "Classification", "Range (m)", "Velocity (m/s)", "Confidence"])
                    # Rows
                    for tr in self.current_tracks:
                        writer.writerow([
                            tr['id'],
                            tr.get('class_label', 'Unknown'),
                            f"{tr['range_m']:.2f}",
                            f"{tr['velocity_m_s']:.2f}",
                            f"{tr.get('confidence', 0.0):.2f}"
                        ])
                print(f"Exported to {filename}")
            except Exception as e:
                print(f"Export failed: {e}")

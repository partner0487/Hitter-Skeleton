"""雙棒八球 8-point 外參：同步拍照、補拍提示、照片匯入與球心提取。"""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import copy
import json
from dataclasses import asdict

from PyQt5.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog,
                            QFormLayout, QGroupBox, QHBoxLayout, QLabel, QSpinBox,
                            QTableWidgetItem, QMessageBox, QPushButton, QHeaderView,
                            QDialog, QDialogButtonBox, QScrollArea, QWidget, QVBoxLayout)
from Src.calibration.calibrate_wood_frame import WoodFrameWindow, ROLES, read_image
from Src.calibration.calibrate import Job
from Src.calibration import calib_store
from Src.calibration.calibration_workflow import save_json
from Src.calibration.two_rod_calibration import (IDS, rod_target, suggest_points,
    coverage_report, fit_two_rod_rig, save_two_rod_rig, DetectionConfig)


class TwoRodWindow(WoodFrameWindow):
    point_ids = {r: IDS for r in ROLES}
    session_schema = 'two_rod_session_v1'
    session_suffix = '_two_rods'
    minimum_pairs = 15
    maximum_pairs = 200

    def target_definition(self):
        return getattr(self, 'active_rod_target', rod_target(
            self.spacing.currentData() if hasattr(self, 'spacing') else 'centers'))

    def _build_ui(self):
        super()._build_ui()
        self.setWindowTitle('雙棒八球 · 8-point 外參校正')
        self.title_label.setText('雙棒八球 · 8-point 外參校正')
        self.steps.setText('1 內參與球距 → 2 拍攝／匯入 15 組以上 → 3 確認球號與補拍 → 4 外參匯出')
        self.geometry_label.setText('兩根棒子各 4 顆螢光粉球，直徑 50 mm。\n'
                                   'A 棒由上而下：A1–A4，間距 200、120、180 mm。\n'
                                   'B 棒由上而下：B1–B4，間距 100、140、210 mm。\n'
                                   '兩台相機都須看見同一組 8 顆球；棒子可各自移動，不需固定兩棒間距。')
        self.target_count.setRange(15, 200)
        self.target_count.setValue(15)
        self.threshold.setValue(2.)
        self.spacing = QComboBox()
        self.spacing.addItem('上述數字是球心到球心距離', 'centers')
        self.spacing.setEnabled(False)  # User confirmed center-to-center distances.
        settings = QGroupBox('球距定義與螢光球提取（沿用 extract_fluorescent_balls）')
        form = QFormLayout(settings)
        form.addRow('距離定義', self.spacing)
        self.detector_fields = {}
        row = QHBoxLayout()
        for name, title, maximum, value in [('hue_min','H 下限',179,135), ('hue_max','H 上限',179,168),
                                           ('saturation_min','S 下限',255,105), ('value_min','V 下限',255,50)]:
            spin = QSpinBox(); spin.setRange(0, maximum); spin.setValue(value)
            self.detector_fields[name] = spin
            row.addWidget(QLabel(title)); row.addWidget(spin)
        form.addRow(row)
        row = QHBoxLayout()
        for name, title, value in [('min_radius_px','最小半徑 px',3), ('min_area_px','最小面積 px²',20),
                                   ('max_area_px','最大面積 px²',2500)]:
            spin = QDoubleSpinBox(); spin.setRange(.1, 1000000); spin.setValue(value)
            self.detector_fields[name] = spin
            row.addWidget(QLabel(title)); row.addWidget(spin)
        form.addRow(row)
        self.setup_page.layout().insertWidget(2, settings)
        # Existing capture/session mechanics remain shared with the wood tool.
        for label in self.pair_page.findChildren(QLabel):
            if 'F1 → F4' in label.text(): label.setText('正面 CF · A1–A4 / B1–B4')
            if 'S1 → S4' in label.text(): label.setText('側面 CS · A1–A4 / B1–B4')
        for button in self.setup_page.findChildren(QPushButton):
            if button.text() == '開始木框校正': button.setText('開始雙棒 8-point 校正')
        self.annotation_controls = QHBoxLayout()
        self.pair_page.layout().insertLayout(self.pair_page.layout().count()-1, self.annotation_controls)
        self.detect_button = self.button('重新提取本組球心', self.redetect_current, self.annotation_controls)
        self.button('調整提取參數', self.edit_detection, self.annotation_controls)
        for role in ROLES:
            for rod in ('A', 'B'):
                self.button('{} 反轉 {} 棒順序'.format(role.upper(), rod),
                            lambda checked=False, r=role, b=rod: self.reverse_rod(r,b), self.annotation_controls)
        self.confirm_button = self.button('確認本組兩視角球號', self.confirm_pair, self.annotation_controls)
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(['組別','資料用途','有效球數','重投影 RMS (px)','棒長 RMSE (mm)'])
        self.result_table.horizontalHeader().setSectionResizeMode(4,QHeaderView.Stretch)
        for button in self.result_page.findChildren(QPushButton):
            if button.text()=='檢視最佳組照片': button.setText('檢視照片與標記')
            if button.text()=='儲存最佳一組外參…': button.setText('匯出 8-point 外參…')
        for label in self.setup_page.findChildren(QLabel):
            if label.text()=='單張重投影 RMS 上限': label.setText('RANSAC 極線誤差門檻')
            if '每一組是同一木框姿態' in label.text():
                label.setText('先完成兩台內參。拍攝時相機、焦距及解析度固定；同組兩棒須停穩，雙機即時拍攝需接 GPIO 同步線。')
        self.result_summary.setText('')
        content=QWidget()
        content.setLayout(self.setup_page.layout())
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(content)
        QVBoxLayout(self.setup_page).addWidget(scroll)
        self.set_setup()

    def set_setup(self):
        super().set_setup()
        self.instruction.setText('選擇兩台既有內參並確認球距定義。準備 15 組以上同步照片，每台至少 15 張。')

    def fail(self,message):
        self.status.setText(message)
        QMessageBox.warning(self,'雙棒 8-point 校正',message)

    def return_to_capture(self):
        super().return_to_capture()
        for i in range(self.annotation_controls.count()):
            self.annotation_controls.itemAt(i).widget().setVisible(False)

    def begin_annotation(self):
        super().begin_annotation()
        if self.phase=='annotate':
            for i in range(self.annotation_controls.count()):
                self.annotation_controls.itemAt(i).widget().setVisible(True)

    def edit_detection(self):
        if self.phase!='annotate' or self.job is not None: return
        dialog=QDialog(self); dialog.setWindowTitle('螢光球提取參數')
        form=QFormLayout(dialog); fields={}
        titles={'hue_min':'H 下限','hue_max':'H 上限','saturation_min':'S 下限','value_min':'V 下限',
                'min_radius_px':'最小半徑 px','min_area_px':'最小面積 px²','max_area_px':'最大面積 px²'}
        for key,value in self.active_detector.items():
            source=self.detector_fields[key]
            spin=QSpinBox() if isinstance(source,QSpinBox) else QDoubleSpinBox()
            spin.setRange(source.minimum(),source.maximum()); spin.setValue(value)
            fields[key]=spin; form.addRow(titles[key],spin)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); form.addRow(buttons)
        if dialog.exec_()==QDialog.Accepted:
            try:
                config=DetectionConfig(**{k:w.value() for k,w in fields.items()}); config.validate()
                previous = self.active_detector
                self.active_detector = asdict(config)
                try:
                    self.persist()
                except (ValueError, OSError):
                    self.active_detector = previous
                    raise
                self.redetect_current()
            except (ValueError,OSError) as exc: self.fail(str(exc))

    def start_session(self):
        try:
            config = DetectionConfig(**{k: field.value() for k,field in self.detector_fields.items()})
            config.validate()
        except ValueError as exc:
            self.fail(str(exc)); return
        self.active_detector = asdict(config)
        self.active_rod_target = rod_target(self.spacing.currentData())
        super().start_session()

    def persist(self):
        if self.session is None: return
        save_json(self.session / 'session.json', {
            'schema': self.session_schema, 'target': self.target_definition(),
            'cameras': self.cameras, 'roles': list(self.roles), 'pairs': self.pairs,
            'target_count': self.active_target_count, 'max_rms_px': self.active_threshold,
            'intrinsic_source': self.intrinsic_source, 'detector': self.active_detector})

    def resume_session(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, '載入雙棒工作階段',
                str(Path(calib_store.CALIB_DIR)/'sessions'), 'JSON (*.json)')
        if not path: return
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                raise ValueError('工作階段 JSON 必須為物件格式。')
            target = rod_target(data['target']['spacing_convention'])
            if data['schema'] != self.session_schema or data['target'] != target or target['spacing_convention']!='centers':
                raise ValueError('不是相符的雙棒工作階段。')
            config = DetectionConfig(**data['detector']); config.validate()
            if not isinstance(data['pairs'], list):
                raise ValueError('工作階段 pairs 必須為陣列。')
            for pair in data['pairs']:
                if not isinstance(pair, dict) or not isinstance(pair.get('confirmed', False), bool):
                    raise ValueError('工作階段的球號確認狀態必須是布林值。')
                for role in ROLES:
                    detection = pair[role].get('detection')
                    if detection is not None and not isinstance(detection, dict):
                        raise ValueError('工作階段的球心提取紀錄格式無效。')
                if pair.get('confirmed') and not self.pair_complete(pair):
                    raise ValueError('已確認的組別必須包含兩視角各 8 個不同球心。')
            if not super().resume_session(path):
                return False
            self.active_detector, self.active_rod_target = asdict(config), target
            self.spacing.setCurrentIndex(self.spacing.findData(target['spacing_convention']))
            for key,value in self.active_detector.items(): self.detector_fields[key].setValue(value)
            return True
        except (ValueError,KeyError,TypeError,AttributeError,OSError) as exc:
            self.fail('讀取失敗：'+str(exc))
            return False

    def update_progress(self):
        super().update_progress()
        self.update_guidance()

    def update_guidance(self):
        report = coverage_report(self.pairs, self.cameras)
        lines = ['已收集 {} 組／目標 {} 組。每組停穩，兩相機全程固定。'.format(len(self.pairs),self.active_target_count)]
        for role in ROLES:
            view = report[role]
            lines.append('{} 缺位置：{}；缺投影角度：{}；{}'.format(
                '正面' if role=='cf' else '側面', '、'.join(view['missing_positions']) or '九宮格已涵蓋',
                '、'.join(view['missing_projected_angles']) or '已涵蓋',
                '請增加遠近／前後傾斜變化' if view['need_distance_variation'] else '影像大小已有變化'))
        lines.append('提示依已提取／標記的球心；投影角度與影像大小不等於真實三維角度／深度。')
        self.instruction.setText('\n'.join(lines))

    def capture_pair(self):
        count = len(self.pairs)
        super().capture_pair()
        if len(self.pairs)>count: self.detect_pairs([len(self.pairs)-1])

    def on_pairs_imported(self, indices):
        self.detect_pairs(indices)

    def detect_pairs(self, indices):
        if self.job is not None or not indices: return
        pairs = copy.deepcopy(self.pairs)
        target, cameras = self.target_definition(), copy.deepcopy(self.cameras)
        config = DetectionConfig(**self.active_detector)
        def work():
            results = {}
            for i in indices:
                results[i] = {}
                for role in ROLES:
                    try:
                        results[i][role] = suggest_points(read_image(pairs[i][role]['path']), cameras[role], target, config)
                    except (ValueError, OSError) as exc:
                        results[i][role] = {'points': {}, 'note': str(exc), 'candidate_count': 0}
            return results
        self.pages.setEnabled(False); self.reset_button.setEnabled(False)
        self.status.setText('正在提取球心、辨認兩根棒子…')
        self.job = Job(work,self)
        def done():
            job,self.job=self.job,None
            self.pages.setEnabled(True); self.reset_button.setEnabled(True)
            try:
                if job.error: raise ValueError(job.error)
                previous = copy.deepcopy(self.pairs)
                for i, views in job.result.items():
                    self.pairs[i]['confirmed']=False
                    for role, detection in views.items():
                        self.pairs[i][role]['points']=detection['points']
                        self.pairs[i][role]['detection']=detection
                try:
                    self.persist()
                except (ValueError, OSError):
                    self.pairs = previous
                    raise
                self.result=None
                if self.phase=='annotate': self.show_pair()
                else: self.update_progress()
                self.status.setText('球心提取完成；請在標記頁確認 A/B 棒與實體上下球號。')
            except (ValueError,OSError) as exc: self.fail(str(exc))
            finally: job.deleteLater()
        self.job.finished.connect(done); self.job.start()

    def redetect_current(self):
        if self.phase=='annotate': self.detect_pairs([self.index])

    def reverse_rod(self,role,rod):
        if self.phase!='annotate' or self.job is not None: return
        points=self.pairs[self.index][role]['points']
        ids=[rod+str(i) for i in range(1,5)]
        if not all(i in points for i in ids): return
        pair = self.pairs[self.index]
        previous = copy.deepcopy(pair)
        values=[points[i] for i in ids]
        points.update(dict(zip(ids,reversed(values))))
        pair['confirmed']=False
        try:
            self.persist()
        except (ValueError, OSError) as exc:
            pair.clear()
            pair.update(previous)
            self.fail('球號順序未保存：' + str(exc))
            return
        self.result=None
        self.show_pair()

    @staticmethod
    def pair_complete(pair):
        """A confirmation requires all eight distinct centers in both views."""
        for role in ROLES:
            points = pair[role]['points']
            if not isinstance(points, dict) or set(points) != set(IDS):
                return False
            if len({tuple(point) for point in points.values()}) != len(IDS):
                return False
        return True

    def confirm_pair(self):
        if self.phase!='annotate' or self.job is not None: return
        pair=self.pairs[self.index]
        if not self.pair_complete(pair):
            self.fail('請先完成兩張照片各 8 個不同的球心。'); return
        previous = copy.deepcopy(pair)
        pair['confirmed']=True
        try: self.persist()
        except (ValueError, OSError) as exc:
            pair.clear()
            pair.update(previous)
            self.fail('球號確認未保存：' + str(exc))
            return
        self.update_annotation_status()

    def mark_point(self,role,x,y):
        if self.phase != 'annotate' or self.job is not None:
            return False
        pair = self.pairs[self.index]
        previous = pair.get('confirmed', False)
        had_confirmation = 'confirmed' in pair
        pair['confirmed'] = False
        if super().mark_point(role, x, y):
            return True
        if had_confirmation:
            pair['confirmed'] = previous
        else:
            pair.pop('confirmed', None)
        return False

    def undo_point(self,role):
        if self.phase != 'annotate' or self.job is not None:
            return False
        pair = self.pairs[self.index]
        previous = pair.get('confirmed', False)
        had_confirmation = 'confirmed' in pair
        pair['confirmed'] = False
        if super().undo_point(role):
            return True
        if had_confirmation:
            pair['confirmed'] = previous
        else:
            pair.pop('confirmed', None)
        return False

    def update_annotation_status(self):
        super().update_annotation_status()
        selected=[p for p in self.pairs if not p.get('excluded')]
        confirmed=sum(p.get('confirmed') is True and self.pair_complete(p) for p in selected)
        pair=self.pairs[self.index]
        self.pair_label.setText('第 {}/{} 組｜正面 {}/8、側面 {}/8｜本組{}｜已確認 {}/{} 組（至少15組）'.format(
            self.index+1,len(self.pairs),len(pair['cf']['points']),len(pair['cs']['points']),
            '已確認' if pair.get('confirmed') else '待確認',confirmed,len(selected)))
        self.calculate_button.setEnabled(len(selected)>=15 and confirmed==len(selected))
        self.update_guidance()
        notes=['選球號後左鍵修改，滾輪放大。A1/B1 是實體上端；請核對兩視角相同球號後確認。']
        for r in ROLES:
            d=pair[r].get('detection',{})
            if d:
                notes.append('{}：{} 顆候選；{}'.format(
                    r.upper(), d.get('candidate_count', '未知'), d.get('note', '')))
        self.status.setText('\n'.join(notes))

    def calculate(self):
        if self.job is not None: return
        pairs=copy.deepcopy([p for p in self.pairs if not p.get('excluded')])
        if len(pairs)<15 or not all(p.get('confirmed') is True and self.pair_complete(p) for p in pairs):
            self.fail('請完成並確認至少 15 組照片。'); return
        cameras,target=copy.deepcopy(self.cameras),copy.deepcopy(self.target_definition())
        threshold=self.active_threshold
        self.pages.setEnabled(False); self.reset_button.setEnabled(False)
        self.status.setText('正在計算 8-point 外參、毫米尺度與保留照片檢查…')
        self.job=Job(lambda:fit_two_rod_rig(cameras,target,pairs,threshold),self)
        self.job.finished.connect(self.calculation_finished); self.job.start()

    def calculation_finished(self):
        job,self.job=self.job,None
        self.pages.setEnabled(True); self.reset_button.setEnabled(True)
        try:
            if job.error: raise ValueError(job.error)
            self.result=job.result
            self.result.update(source_session=str(self.session),camera_serials=dict(zip(ROLES,self.roles)))
            save_two_rod_rig(self.session/'stereo.json',self.result)
            self.phase='result'; self.pages.setCurrentWidget(self.result_page)
            self.instruction.setText('完成：多組照片共同估計外參；保留照片未參與外參與尺度擬合。')
            self.result_summary.setText('雙機基線 {:.1f} mm｜訓練 {} 組｜保留驗證 {} 組'.format(
                self.result['baseline_mm'],len(self.result['training_pairs']),len(self.result['held_out_pairs'])))
            rows=self.result['candidate_diagnostics']; self.result_table.setRowCount(len(rows))
            for i,d in enumerate(rows):
                values=[str(next(j+1 for j,p in enumerate(self.pairs) if p['id']==d['id'])),
                        ('保留驗證' if d['held_out'] else '訓練')+(' 通過' if d['accepted'] else ' 未通過'),
                        str(d['valid_points']),'{:.3f}'.format(d['reprojection_rms_px']),
                        '—' if d['length_rmse_mm'] is None else '{:.2f}'.format(d['length_rmse_mm'])]
                for c,value in enumerate(values): self.result_table.setItem(i,c,QTableWidgetItem(value))
            self.result_text.setPlainText('\n'.join(self.result['warnings']))
            self.status.setText('已保存工作階段結果；請匯出 stereo.json 供 3D 的 8Points 模式讀取。')
        except (ValueError,OSError) as exc:
            self.result=None; self.fail(str(exc))
        finally: job.deleteLater()

    def show_best_pair(self):
        self.begin_annotation()

    def export_result(self):
        if self.result is None: return
        path,_=QFileDialog.getSaveFileName(self,'匯出雙棒 8-point 外參',
            str(Path(calib_store.CALIB_DIR)/'two_rods'/'stereo.json'),'JSON (*.json)')
        if path:
            try:
                save_two_rod_rig(path,self.result)
                self.status.setText('已匯出：'+path)
            except (ValueError,OSError) as exc: self.fail(str(exc))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.parse_args(argv)
    try:
        from Src.cameras.single_camera.flir_camera_system import FlirCameraSystem
        from Src.cameras.two_cameras.dual_flir_system import DualFlirSystem
    except ImportError:
        pass
    app=QApplication.instance() or QApplication(sys.argv)
    window=TwoRodWindow(); window.show()
    return app.exec_()


if __name__=='__main__': sys.exit(main())

"""Cancellable photo staging; commit session metadata on the GUI thread only."""
import copy
from pathlib import Path
from uuid import uuid4

from PyQt5.QtCore import QThread, pyqtSignal

from Src.calibration.calibration_workflow import match_image_pairs


class ImportCancelled(Exception):
    pass


def remove_staged_images(paths):
    """Remove only this import's generated files, including partial copies."""
    errors = []
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:
            errors.append(str(exc))
    if errors:
        raise OSError("無法清理匯入暫存照片：" + "; ".join(errors))


def stage_pairs(folder, session, cameras, read_image, cancelled=lambda: False,
                progress=lambda completed, total, name: None):
    """Copy and validate without changing the active session or keeping images in RAM."""
    names = match_image_pairs(folder, minimum=1)
    pending, written = [], []
    batch = uuid4().hex

    def check_cancelled():
        if cancelled():
            raise ImportCancelled()

    try:
        for index, name in enumerate(names):
            pair = {"id": "{}_import_{:03d}".format(batch, index + 1), "original_name": name}
            for role_index, role in enumerate(("cf", "cs")):
                check_cancelled()
                source = Path(folder) / role / name
                dest = (Path(session) / role / (pair["id"] + source.suffix.lower())).resolve()
                # Exclusive creation prevents overwriting any existing session file.
                with dest.open("xb") as target:
                    written.append(dest)
                    with source.open("rb") as origin:
                        while True:
                            check_cancelled()
                            chunk = origin.read(1024 * 1024)
                            if not chunk:
                                break
                            target.write(chunk)
                # Validate the saved bytes, even if the source changed during import.
                image = read_image(dest)
                size = [image.shape[1], image.shape[0]]
                del image
                if size != list(cameras[role]["image_size"]):
                    raise ValueError("{} 的 {} 照片尺寸 {} 與內參 {} 不符。".format(
                        name, role.upper(), size, cameras[role]["image_size"]))
                pair[role] = {"points": {}, "image_size": size, "path": str(dest),
                              "source_path": str(source.resolve())}
                progress(index * 2 + role_index + 1, len(names) * 2, name)
            pending.append(pair)
        check_cancelled()
        return pending
    except Exception:
        remove_staged_images(written)
        raise


class PairImportWorker(QThread):
    progress = pyqtSignal(int, int, str)

    def __init__(self, folder, session, cameras, read_image, parent=None):
        super().__init__(parent)
        self.folder, self.session = folder, session
        self.cameras, self.read_image = copy.deepcopy(cameras), read_image
        self.result, self.error, self.cancelled = None, None, False

    def cancel(self):
        # QThread clears its interruption flag when run() returns. Keep the
        # user's request until the GUI has decided whether to commit results.
        self.cancelled = True
        self.requestInterruption()

    def run(self):
        try:
            self.result = stage_pairs(self.folder, self.session, self.cameras, self.read_image,
                                      lambda: self.cancelled or self.isInterruptionRequested(),
                                      self.progress.emit)
        except ImportCancelled:
            self.cancelled = True
        except Exception as exc:
            self.error = str(exc)

"""tatpulsar.gui.tempo_gui
==========================

A PyQt-based graphical front-end that wraps the pulsar timing utilities
provided by ``tatpulsar`` and `PINT <https://nanograv-pint.readthedocs.io>`_.

The application focuses on a TEMPO2-like workflow:

* Load pulsar ephemerides (``.par``) and time-of-arrival files (``.tim``).
* Inspect and interactively (de)select individual TOAs.
* Choose which timing parameters are free to vary in a weighted least-squares fit.
* Execute the fit using :class:`pint.fitter.WLSFitter` and visualise timing
  residuals in real time.
* Export the curated TOA set and the updated timing model for further analysis.

The module exposes a ``main()`` function so that it can be used as a console
entry-point (e.g. ``tatpulsar-tempo-gui``).
"""

from __future__ import annotations

import copy
import dataclasses
import io
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from astropy import units as u
from astropy.units import UnitConversionError

from tatpulsar.pulse import residuals as residual_utils

try:
    from PyQt5 import QtCore, QtWidgets
    from PyQt5.QtCore import Qt
except ImportError as exc:  # pragma: no cover - only triggered when PyQt5 missing
    raise ImportError(
        "PyQt5 is required to use the tatpulsar timing analysis GUI."
    ) from exc

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import pint.toa as pint_toa
from pint.fitter import WLSFitter
from pint.models import get_model
from pint.residuals import Residuals
from pint.toa import TOAs


@dataclasses.dataclass
class FitResult:
    """Container holding the outcome of the latest timing fit."""

    mjds: np.ndarray
    residuals_usec: np.ndarray
    residuals_sec: np.ndarray
    residual_errors_usec: np.ndarray
    wrms_usec: float
    fitter: Optional[WLSFitter] = None


class ResidualsCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas embedded inside the Qt GUI."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._figure = Figure(figsize=(6, 4))
        self._axes = self._figure.add_subplot(111)
        super().__init__(self._figure)
        self.setParent(parent)
        self._axes.set_xlabel("MJD")
        self._axes.set_ylabel("Residual (µs)")
        self._axes.grid(True, ls="--", alpha=0.3)

    def plot(self, result: FitResult) -> None:
        self._axes.clear()
        self._axes.set_xlabel("MJD")
        self._axes.set_ylabel("Residual (µs)")
        if result.mjds.size:
            self._axes.errorbar(
                result.mjds,
                result.residuals_usec,
                yerr=result.residual_errors_usec,
                fmt="o",
                ms=4,
                lw=1,
                ecolor="#888888",
                mec="#1f77b4",
                mfc="#1f77b4",
            )
            ylim = np.nanmax(np.abs(result.residuals_usec))
            if not np.isfinite(ylim) or ylim == 0:
                ylim = 1
            self._axes.set_ylim(-1.2 * ylim, 1.2 * ylim)
        self._axes.grid(True, ls="--", alpha=0.3)
        self.draw_idle()


class ToaTableModel(QtCore.QAbstractTableModel):
    """Qt model exposing a TOA table with checkboxes for selection."""

    COLUMNS = (
        "Use",
        "MJD",
        "Error (µs)",
        "Site",
        "Flags",
    )

    def __init__(
        self,
        session: TimingSession,
        parent: Optional[QtWidgets.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._toas: Optional[TOAs] = None
        self._mask: Optional[np.ndarray] = None

    # -- lifecycle ---------------------------------------------------------
    def set_toas(self, toas: Optional[TOAs], mask: np.ndarray) -> None:
        self.beginResetModel()
        self._toas = toas
        self._mask = mask
        self.endResetModel()

    # -- Qt model API ------------------------------------------------------
    def rowCount(self, _parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:  # noqa: N802 (Qt API)
        return 0 if self._toas is None else len(self._toas.table)

    def columnCount(
        self, _parent: QtCore.QModelIndex = QtCore.QModelIndex()
    ) -> int:  # noqa: N802 (Qt API)
        return len(self.COLUMNS)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = Qt.DisplayRole,
    ):  # noqa: D401
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.COLUMNS[section]
        return section + 1

    def data(self, index: QtCore.QModelIndex, role: int = Qt.DisplayRole):  # noqa: D401
        if not index.isValid() or self._toas is None or self._mask is None:
            return None
        row = index.row()
        col = index.column()
        if col == 0:
            if role == Qt.CheckStateRole:
                return Qt.Checked if self._mask[row] else Qt.Unchecked
            if role in (Qt.DisplayRole, Qt.EditRole):
                return ""
            return None

        mjds = self._toas.get_mjds().value
        errs = self._toas.get_errors().to(u.microsecond).value
        site = self._toas.table["obs"] if "obs" in self._toas.table.colnames else None
        flag_strings: Sequence[str] = []
        if "flags" in self._toas.table.colnames:
            row_flags = self._toas.table["flags"][row]
            if isinstance(row_flags, dict):
                flag_strings = [f"{k}={v}" for k, v in row_flags.items()]
            else:
                flag_strings = [str(row_flags)]
        if role in (Qt.DisplayRole, Qt.EditRole):
            if col == 1:
                return f"{mjds[row]:.10f}"
            if col == 2:
                return f"{errs[row]:.3f}"
            if col == 3:
                return str(site[row]) if site is not None else ""
            if col == 4:
                return " ".join(flag_strings)
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:  # noqa: D401
        if not index.isValid():
            return Qt.NoItemFlags
        base = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if index.column() == 0:
            base |= Qt.ItemIsUserCheckable
        return base

    def setData(
        self,
        index: QtCore.QModelIndex,
        value,
        role: int = Qt.EditRole,
    ) -> bool:
        if (
            self._mask is None
            or not index.isValid()
            or index.column() != 0
            or role not in (Qt.EditRole, Qt.CheckStateRole)
        ):
            return False
        use_toa = value == Qt.Checked
        self._session.mark_toa(index.row(), use_toa)
        self.dataChanged.emit(index, index, [Qt.CheckStateRole])
        return True

    # -- convenience -------------------------------------------------------
    def mask(self) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("TOA table mask accessed before model initialisation")
        return self._mask


class ParameterSelectionWidget(QtWidgets.QWidget):
    """Checklist of timing parameters that may be freed during the fit."""

    selectionChanged = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        self._value_labels: Dict[str, QtWidgets.QLabel] = {}

        btn_layout = QtWidgets.QHBoxLayout()
        self._layout.addLayout(btn_layout)

        select_all = QtWidgets.QPushButton("Select all")
        select_all.clicked.connect(self._select_all)
        btn_layout.addWidget(select_all)

        select_none = QtWidgets.QPushButton("Select none")
        select_none.clicked.connect(self._select_none)
        btn_layout.addWidget(select_none)

        btn_layout.addStretch(1)

        self._params_container = QtWidgets.QVBoxLayout()
        self._layout.addLayout(self._params_container)
        self._layout.addStretch(1)

    def set_model(self, model) -> None:
        # clear existing widgets
        while self._params_container.count():
            child = self._params_container.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()
        self._checkboxes.clear()
        self._value_labels.clear()

        free_params = getattr(model, "free_params", [])
        if not free_params:
            label = QtWidgets.QLabel("No free parameters available")
            self._params_container.addWidget(label)
            return

        for par_name in free_params:
            param = getattr(model, par_name)
            checkbox = QtWidgets.QCheckBox(par_name)
            checkbox.setChecked(not getattr(param, "frozen", False))
            checkbox.stateChanged.connect(self.selectionChanged.emit)

            value_label = QtWidgets.QLabel(self._format_param_value(param))
            value_label.setToolTip(param.as_parfile_line())

            row = QtWidgets.QHBoxLayout()
            row.addWidget(checkbox)
            row.addStretch(1)
            row.addWidget(value_label)

            container = QtWidgets.QWidget()
            container.setLayout(row)
            self._params_container.addWidget(container)

            self._checkboxes[par_name] = checkbox
            self._value_labels[par_name] = value_label

    @staticmethod
    def _format_param_value(param) -> str:
        if hasattr(param, "uncertainty") and param.uncertainty is not None:
            return f"{param.value:.6g} ± {param.uncertainty:.2g}"
        return f"{param.value:.6g}"

    def selected_parameters(self) -> List[str]:
        return [name for name, box in self._checkboxes.items() if box.isChecked()]

    def refresh_values(self, model) -> None:
        for name, label in self._value_labels.items():
            param = getattr(model, name)
            label.setText(self._format_param_value(param))
            label.setToolTip(param.as_parfile_line())
        for name, box in self._checkboxes.items():
            box.blockSignals(True)
            box.setChecked(not getattr(getattr(model, name), "frozen", False))
            box.blockSignals(False)

    def _select_all(self) -> None:
        for box in self._checkboxes.values():
            box.setChecked(True)
        self.selectionChanged.emit()

    def _select_none(self) -> None:
        for box in self._checkboxes.values():
            box.setChecked(False)
        self.selectionChanged.emit()


class TimingSession(QtCore.QObject):
    """Data/logic layer shared between widgets."""

    stateChanged = QtCore.pyqtSignal()
    fitCompleted = QtCore.pyqtSignal(FitResult)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.model = None
        self.toas: Optional[TOAs] = None
        self._mask: Optional[np.ndarray] = None
        self._last_fit: Optional[FitResult] = None
        self.current_parfile: Optional[Path] = None
        self.current_timfile: Optional[Path] = None

    # -- data access -------------------------------------------------------
    def mask(self) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("TOA mask requested before files were loaded")
        return self._mask

    def set_mask(self, mask: np.ndarray) -> None:
        if self._mask is None or mask.shape != self._mask.shape:
            raise ValueError("Mask shape mismatch")
        self._mask[:] = mask
        self.stateChanged.emit()

    def mark_toa(self, index: int, use: bool) -> None:
        mask = self.mask()
        if 0 <= index < mask.size:
            mask[index] = use
            self.stateChanged.emit()

    # -- loading -----------------------------------------------------------
    def load_files(self, par_path: Path, tim_path: Path) -> None:
        self.model = get_model(str(par_path))
        self.toas = pint_toa.get_TOAs(str(tim_path), usepickle=False)
        self._mask = np.ones(self.toas.ntoas, dtype=bool)
        self.current_parfile = Path(par_path)
        self.current_timfile = Path(tim_path)
        # Reset existing fit summary
        self._last_fit = None
        self.stateChanged.emit()

    # -- computation -------------------------------------------------------
    def selected_toas(self) -> TOAs:
        if self.toas is None or self._mask is None:
            raise RuntimeError("TOAs requested before files were loaded")
        if np.all(self._mask):
            return self.toas
        subset = copy.deepcopy(self.toas)
        subset.select(self._mask)  # type: ignore[attr-defined]
        return subset

    def run_fit(self, free_parameters: Iterable[str]) -> FitResult:
        if self.model is None or self.toas is None:
            raise RuntimeError("Cannot run fit before data are loaded")

        free_parameters = list(free_parameters)
        model_to_fit = copy.deepcopy(self.model)

        for param_name in getattr(model_to_fit, "free_params", []):
            getattr(model_to_fit, param_name).frozen = param_name not in free_parameters

        work_toas = self.selected_toas()
        fitter = WLSFitter(work_toas, model_to_fit)
        fitter.fit_toas()

        resids: Residuals = fitter.resids
        residuals_sec = resids.time_resids.to(u.second).value
        residuals_usec = resids.time_resids.to(u.microsecond).value
        toa_errors_usec = work_toas.get_errors().to(u.microsecond).value
        toa_errors_sec = work_toas.get_errors().to(u.second).value
        mjds = work_toas.get_mjds().value

        wrms_sec = residual_utils.rms(residuals_sec, toa_errors_sec)
        wrms_usec = wrms_sec * 1e6

        result = FitResult(
            mjds=mjds,
            residuals_usec=residuals_usec,
            residuals_sec=residuals_sec,
            residual_errors_usec=toa_errors_usec,
            wrms_usec=wrms_usec,
            fitter=fitter,
        )

        # Persist updates back to session
        self.model = fitter.model
        self._last_fit = result
        self.stateChanged.emit()
        self.fitCompleted.emit(result)
        return result

    # -- exports -----------------------------------------------------------
    def export_par(self, destination: Path) -> None:
        if self.model is None:
            raise RuntimeError("No timing model loaded")
        destination = Path(destination)
        destination.write_text(self.model.as_parfile(), encoding="utf-8")

    def export_tim(self, destination: Path) -> None:
        if self.toas is None or self._mask is None:
            raise RuntimeError("No TOAs loaded")
        selected_indices, = np.where(self._mask)
        buffer = io.StringIO()
        buffer.write("FORMAT 1\n")
        table = self.toas.table
        colnames = set(table.colnames)

        for idx in selected_indices:
            row = table[idx]

            mjd_col = row["mjd"]
            mjd_val = float(mjd_col.value) if hasattr(mjd_col, "value") else float(mjd_col)

            err_val = None
            if "error" in colnames:
                err_col = row["error"]
                if hasattr(err_col, "to"):
                    err_val = err_col.to(u.microsecond).value
                else:
                    err_val = float(err_col)
            if err_val is None:
                err_val = 0.0

            freq_val = 0.0
            if "freq" in colnames:
                freq_col = row["freq"]
                if hasattr(freq_col, "to"):
                    try:
                        freq_val = float(freq_col.to(u.MHz).value)
                    except UnitConversionError:
                        freq_val = float(freq_col.to(u.Hz).value / 1e6)
                else:
                    freq_val = float(freq_col)

            flags_obj = row["flags"] if "flags" in colnames else {}
            if isinstance(flags_obj, dict):
                flag_tokens = [f"-{k} {v}" for k, v in flags_obj.items()]
            else:
                flag_tokens = [] if not flags_obj else [str(flags_obj)]
            flag_str = " ".join(flag_tokens)

            obs_code = str(row["obs"]).strip()
            buffer.write(
                f"{obs_code:<4s} {freq_val:>12.6f} {mjd_val:.15f} {err_val:>9.3f} {flag_str}\n"
            )
        destination.write_text(buffer.getvalue(), encoding="utf-8")

    # -- status -----------------------------------------------------------
    def last_fit(self) -> Optional[FitResult]:
        return self._last_fit


class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("tatpulsar Timing Analysis")
        self.resize(1200, 720)

        self._session = TimingSession()

        # Central splitter: left table, right controls
        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(6, 6, 6, 6)
        central_layout.setSpacing(6)
        self.setCentralWidget(central)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        central_layout.addWidget(splitter, stretch=3)

        # TOA table -------------------------------------------------------
        self._toa_model = ToaTableModel(self._session, self)
        self._toa_view = QtWidgets.QTableView()
        self._toa_view.setModel(self._toa_model)
        self._toa_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        splitter.addWidget(self._toa_view)

        # Control panel ---------------------------------------------------
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setContentsMargins(4, 4, 4, 4)
        control_layout.setSpacing(8)
        splitter.addWidget(control_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self._param_widget = ParameterSelectionWidget()
        self._param_widget.selectionChanged.connect(self._on_parameter_selection_changed)
        control_layout.addWidget(QtWidgets.QLabel("Fit parameters"))
        control_layout.addWidget(self._param_widget, stretch=1)

        self._fit_button = QtWidgets.QPushButton("Run fit")
        self._fit_button.clicked.connect(self._on_run_fit)
        control_layout.addWidget(self._fit_button)

        self._export_par_button = QtWidgets.QPushButton("Save updated par…")
        self._export_par_button.clicked.connect(self._on_export_par)
        control_layout.addWidget(self._export_par_button)

        self._export_tim_button = QtWidgets.QPushButton("Save selected TOAs…")
        self._export_tim_button.clicked.connect(self._on_export_tim)
        control_layout.addWidget(self._export_tim_button)

        self._summary_label = QtWidgets.QLabel("Load files to begin timing analysis.")
        self._summary_label.setWordWrap(True)
        control_layout.addWidget(self._summary_label)
        control_layout.addStretch(1)

        # Residual plot ---------------------------------------------------
        self._canvas = ResidualsCanvas()
        central_layout.addWidget(self._canvas, stretch=2)

        # connections -----------------------------------------------------
        self._session.stateChanged.connect(self._refresh_state)
        self._session.fitCompleted.connect(self._on_fit_completed)

        # Menus -----------------------------------------------------------
        file_menu = self.menuBar().addMenu("&File")

        load_action = QtWidgets.QAction("Load par/tim…", self)
        load_action.triggered.connect(self._on_load_files)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        quit_action = QtWidgets.QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    # -- actions ----------------------------------------------------------
    def _on_load_files(self) -> None:
        par_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select TEMPO2 par file",
            str(self._session.current_parfile or Path.cwd()),
            "Par files (*.par);;All files (*)",
        )
        if not par_path:
            return
        tim_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select TEMPO2 tim file",
            str(self._session.current_timfile or Path(par_path).parent),
            "ToA files (*.tim);;All files (*)",
        )
        if not tim_path:
            return
        try:
            self._session.load_files(Path(par_path), Path(tim_path))
        except Exception as exc:  # pragma: no cover - GUI feedback path
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to load files",
                f"An error occurred while loading timing files:\n\n{exc}",
            )

    def _on_run_fit(self) -> None:
        if self._session.model is None:
            QtWidgets.QMessageBox.information(
                self,
                "No data loaded",
                "Load a par/tim pair before running a fit.",
            )
            return
        free_params = self._param_widget.selected_parameters()
        if not free_params:
            answer = QtWidgets.QMessageBox.question(
                self,
                "No parameters selected",
                "No parameters are currently selected for fitting.\n"
                "Run a residual evaluation with the model frozen?",
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        try:
            result = self._session.run_fit(free_params)
            self._canvas.plot(result)
        except Exception as exc:  # pragma: no cover - GUI feedback path
            QtWidgets.QMessageBox.critical(
                self,
                "Fit failed",
                f"The timing fit did not complete successfully:\n\n{exc}",
            )

    def _on_export_par(self) -> None:
        if self._session.model is None:
            return
        default_dir = (
            self._session.current_parfile.parent
            if self._session.current_parfile is not None
            else Path.cwd()
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save updated par file",
            str(default_dir / "updated.par"),
            "Par files (*.par);;All files (*)",
        )
        if not path:
            return
        try:
            self._session.export_par(Path(path))
        except Exception as exc:  # pragma: no cover - GUI feedback path
            QtWidgets.QMessageBox.critical(
                self,
                "Export failed",
                f"Could not save par file:\n\n{exc}",
            )

    def _on_export_tim(self) -> None:
        if self._session.toas is None:
            return
        default_dir = (
            self._session.current_timfile.parent
            if self._session.current_timfile is not None
            else Path.cwd()
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save selected TOAs",
            str(default_dir / "selected.tim"),
            "ToA files (*.tim);;All files (*)",
        )
        if not path:
            return
        try:
            self._session.export_tim(Path(path))
        except Exception as exc:  # pragma: no cover - GUI feedback path
            QtWidgets.QMessageBox.critical(
                self,
                "Export failed",
                f"Could not save TOAs:\n\n{exc}",
            )

    # -- callbacks --------------------------------------------------------
    def _refresh_state(self) -> None:
        if self._session.toas is None:
            self._toa_model.set_toas(None, np.zeros(0, dtype=bool))
            self._summary_label.setText("Load files to begin timing analysis.")
            return

        self._toa_model.set_toas(self._session.toas, self._session.mask())
        self._param_widget.set_model(self._session.model)

        n_use = int(self._session.mask().sum())
        self._summary_label.setText(
            f"Loaded {self._session.toas.ntoas} TOAs – {n_use} selected for fitting."
        )

    def _on_parameter_selection_changed(self) -> None:
        # Keep summary text focused on data; parameter selections simply
        # update the info label if a fit has already been performed.
        fit = self._session.last_fit()
        if fit is not None:
            self._update_summary_with_fit(fit)

    def _on_fit_completed(self, result: FitResult) -> None:
        self._param_widget.refresh_values(self._session.model)
        self._update_summary_with_fit(result)

    def _update_summary_with_fit(self, result: FitResult) -> None:
        lines = [
            f"Residual WRMS: {result.wrms_usec:.2f} µs",
            f"Active TOAs: {result.mjds.size}",
        ]
        self._summary_label.setText("\n".join(lines))


def main() -> None:
    """Launch the GUI application."""

    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


__all__ = [
    "MainWindow",
    "main",
]

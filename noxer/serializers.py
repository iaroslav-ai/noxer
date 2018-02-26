import os
import pickle as pc


class ExperimentSerializer():
    """Helper class to serialize experiment results,
    and to save figures. The folder structure is managed
    by this class. All the objects are pickled to the
    self.pc_folder folder. All the figures are saved to
    self.fig_folder."""
    def __init__(self, folder):
        self.folder = folder
        self.pc_folder = os.path.join(folder, 'pickle')
        self.fig_folder = os.path.join(folder, 'figures')
        self.context = ""

    def _file_name(self, obj_name):
        """Get file name for pickle file with particular context."""
        filepath = os.path.join(
            self.pc_folder, self.context+"_"+obj_name+".pc"
        )
        return filepath

    def _fig_file_name(self, fig_name):
        """Get file name for figure file with particular context."""
        filepath = os.path.join(
            self.fig_folder, self.context +"_" + fig_name
        )
        return filepath

    def __setitem__(self, obj_name, obj):
        """Pickle object under some name `obj_name`."""
        if not os.path.exists(self.pc_folder):
            os.makedirs(self.pc_folder)
        fname = self._file_name(obj_name)
        pc.dump(obj, open(fname, 'wb'))

    def __getitem__(self, obj_name):
        """Deserialize the object under `obj_name` name."""
        fname = self._file_name(obj_name)
        return pc.load(open(fname, 'rb'))

    def savefig(self, ax, name):
        """Save the content of axis / plot under some name."""
        if not os.path.exists(self.fig_folder):
            os.makedirs(self.fig_folder)
        fname = self._fig_file_name(name)
        ax.savefig(fname)
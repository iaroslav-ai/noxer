import os
import pickle as pc
import csv
from collections import OrderedDict

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


class FolderDatasetReader():
    """A class that can be used to
    read dataset from a folder.
    The following dataset structures are supported at the moment:
    ________________________________________________________
    ID="SIMPLE_CLASIF_FOLDERS"
    Classification dataset with folder names as categories:
    root folder
    |-folder of class a, named after class
    |-|-element 1
    |-|-element 2
    |-|-...
    |-folder of class b, named after class
    |-...
    _________________________________________________________
    ID="CSV_DATA_FOLDER"
    Arbitrary dataset, with csv file specifying data elements:
    root folder
    |-dataset.csv
    |- ... arbitrary folder structure

    dataset.csv contains following keys:
    * X_file_n, n is a string: root relative paths of input files
    * X_n, n is a string, n != "_file": input value, as category or number
    * Y_file_n, n is a string: root relative paths of output files
    * Y_n, n is a string, n != "_file": output value, as category or number
    * partition, optional: describes to which partition a training point belongs to

    """

    folder_dataset = "SIMPLE_CLASIF_FOLDERS"
    csv_folder_dataset = "SIMPLE_CLASIF_FOLDERS"
    csv_folder_name = "dataset.csv"

    supported_formats = {'json', 'wav', 'jpg', 'jpeg', 'png'}

    image_file = 'image'
    audio_file = 'audio'
    json_file = 'json'

    def __init__(self, root_folder, point_preprocess=None):
        self.root_folder = root_folder
        self.dataset_type = None
        self.point_preprocess = point_preprocess
        self.reader = None
        self.reader_resetter = None

    def _read_folder_data(self):
        categories = os.listdir(self.root_folder)
        R = []

        # read data for every category
        for c in categories:
            cat_path = os.path.join(self.root_folder, c)

            for f in os.listdir(cat_path):
                if not f.split('.')[-1] in self.supported_formats:
                    print("Skipped %s" % f)
                    continue

                # read the wav file
                R.append(
                    OrderedDict([
                        ("X_file", os.path.join(c, f)),
                        ("Y", c)
                    ]))


        return R

    def determine_reader(self):
        """Detect the structure of the dataset"""
        csv_path = os.path.join(self.root_folder, self.csv_folder_name)

        # check if csv file exists
        if os.path.exists(csv_path):
            self.dataset_type = self.csv_folder_dataset
            self.reader_resetter = lambda: csv.DictReader(open(csv_path, 'r'))
            return

        self.dataset_type = self.folder_dataset
        # read contents
        data = self._read_folder_data()
        self.reader_resetter = lambda: iter(data)

    def read_file(self, rel_path):

        path = os.path.join(self.root_folder, rel_path)
        ext = path.split('.')[-1].lower()

        # Image files
        if ext in {'jpg', 'png', 'jpeg'}:
            from scipy.misc import imread
            info = {"type": self.image_file}
            result = imread(path)
            return info, result

        # Wav files
        if ext in {'wav'}:
            import wavefile
            info = {"type": self.audio_file}
            result = wavefile.load(path)[1]
            return info, result

        # JSON files
        if ext in {'json'}:
            import json
            info = {"type": self.json_file}
            with open(path, 'r') as f:
                result = json.load(f)
            return info, result

    def read(self, n=-1, partition="train"):
        """
        Reads part of the dataset.

        Parameters:
        ----------
        n: int
            Size of the data chunk to read.

        partition: string
            What particular data partition to read.
            This option is ignored if not partition is
            specified.

        Returns:
        --------
        result: tuple of lists
            A list of all groups in the dataset. Normally
            these are inputs and outputs, but can be extended
            further.

        """
        groups = OrderedDict()

        # initialize if not done so already
        if self.reader_resetter is None:
            self.determine_reader()

        # reset reader if necessary
        if self.reader is None:
            self.reader = self.reader_resetter()

        while n != 0:
            try:
                row = next(self.reader)
            except StopIteration:
                # break if finished reading
                break

            # add the groups if necessary
            first_letters = set([v[0] for v in row.keys() if not v[0].islower()])

            for l in sorted(first_letters):
                if not l in groups:
                    groups[l] = []

            # separate all data into inputs and outputs
            for group in groups:
                # select all elements that correspond to this group
                elements = OrderedDict([
                    (k, v) for k, v in row.items() if k.startswith(group)
                ])

                # select all files
                files = OrderedDict([
                    (k, v) for k, v in elements.items() if k.startswith(group+"_file")
                ])

                # convert to array in a sorted manner
                files = list(self.read_file(v)[1] for v in files.values())

                # select all other elements
                other = OrderedDict([
                    (k, v) for k, v in elements.items() if not k.startswith(group + "_file")
                ])
                other = list(other.values())

                total_result = files+other

                # squeeze the group if necessary
                if len(total_result) == 1:
                    total_result = total_result[0]

                groups[group].append(total_result)

        # convert group to outputs
        result = tuple(groups.values())

        return result
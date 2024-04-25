
from project.utils.file_system import create_dir_if_not_exists
from project.constants import UI_PRMD_DIR


class KimoreDataPreparation:
    def __init__(self):
        create_dir_if_not_exists(UI_PRMD_DIR)

    # def prepare_data(self):

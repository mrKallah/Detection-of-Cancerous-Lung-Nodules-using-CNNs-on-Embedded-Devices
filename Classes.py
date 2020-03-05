

class TagAttribute:
    """
    A class for storing the tag attributes for xml files
    """
    def __init__(self, id, value):
        self.ID = id
        self.Value = value

    def ID(self):
        """
        An id of the patient, used for matching xml data and dcm data
        :return: returns the id
        """
        return "{}".format(self.ID)

    def value(self):
        """
        The value of the tag
        :return: returns the value
        """
        return "{}".format(self.Value)

    def tostring(self):
        """
        generates a string from the value and ID for the patient
        :return: a string containing value and ID for the patient
        """
        return "{} | {}".format(self.ID, self.Value)


class Nodule:
    """
    A class for storing information about nodules
    """
    subtlety = 0
    internalStructure = 0
    calcification = 0
    sphericity = 0
    margin = 0
    lobulation = 0
    spiculation = 0
    texture = 0
    malignancy = 0
    imageZposition = 0
    imageSOP_UID = 0
    NoduleClass = 0
    SliceLocation = None


class DataPoint:
    """
    A class for storing the tags, dcm file information and nodule type
    """
    def __init__(self, tags, dcm, is_nodule):
        self.tags = tags
        self.dcm = dcm
        self.is_nodule = is_nodule


class Output:
    """
    The final output and class
    """
    def __init__(self, pixel_array, nodule_class, malignancy):
        self.pixel_array = pixel_array
        self.nodule_class = nodule_class
        self.malignancy = malignancy

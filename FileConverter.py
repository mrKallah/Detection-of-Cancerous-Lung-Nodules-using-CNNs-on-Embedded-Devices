import pydicom

import xml.etree.ElementTree as ET
from anytree import Node, RenderTree
from Classes import TagAttribute
from Classes import Nodule
from Classes import DataPoint
from Classes import Output
from Utils import *
from anytree.search import findall

verbose = False


def read_dcm_file(filename):
    """
    uses pydicom to read a dcm file. Kept in method so if in future changing away from pydicom should be easy
    :param filename: the filename of the dcm file to read
    :return: reruns the contents of the read files
    """
    dataset = pydicom.dcmread(filename)
    return dataset


def get_attributes(child, root):
    """
    recursively looks through a XML file and formats it as a node based tree
    :param child:
    :param root:
    :return: the sturctured tree
    """
    try:
        for c in child:
            tree_child = TagAttribute(c.tag.replace("{http://www.nih.gov}", ""), c.text)
            tree_child = Node(tree_child, root)
            get_attributes(c, tree_child)
    except:
        return root


def extract_nodule_information(n, nodule_class):
    """
    searches the tree for the wanted information to extract and stores it in a new variable
    :param n: the node to extract information from
    :param nodule_class: the already extracted nodule class
    :return: the nodule class with information about the nodule in that scan slice
    """

    def convert_xml_values_to_integers(value, found):
        """
        looks for a value in the xml subheaders and returns its value and whether its been found or not
        :param value: value to search for
        :param found: a value used for multiple conversions to see if any of the values have been found.
        :return:
        """
        if value != ():
            arr = []
            for s in value:
                arr.append(s.name.Value)
                found = True

            if len(arr) == 1:
                nod = arr[0]
            else:
                nod = arr
            return nod, found
        return 0, found

    # searches for the xml headers in the file
    subtlety = findall(n, filter_=lambda node: node.name.ID in ("subtlety"))
    internalStructure = findall(n, filter_=lambda node: node.name.ID in ("internalStructure"))
    calcification = findall(n, filter_=lambda node: node.name.ID in ("calcification"))
    sphericity = findall(n, filter_=lambda node: node.name.ID in ("sphericity"))
    margin = findall(n, filter_=lambda node: node.name.ID in ("margin"))
    lobulation = findall(n, filter_=lambda node: node.name.ID in ("lobulation"))
    spiculation = findall(n, filter_=lambda node: node.name.ID in ("spiculation"))
    texture = findall(n, filter_=lambda node: node.name.ID in ("texture"))
    malignancy = findall(n, filter_=lambda node: node.name.ID in ("malignancy"))
    imageZposition = findall(n, filter_=lambda node: node.name.ID in ("imageZposition"))
    imageSOP_UID = findall(n, filter_=lambda node: node.name.ID in ("imageSOP_UID"))
    SliceLocation = findall(n, filter_=lambda node: node.name.ID in ("imageZposition"))


    nodule = Nodule()

    # converts the xml files to the nodule class
    found = False
    nodule.subtlety, found = convert_xml_values_to_integers(subtlety, found)
    nodule.internalStructure, found = convert_xml_values_to_integers(internalStructure, found)
    nodule.calcification, found = convert_xml_values_to_integers(calcification, found)
    nodule.sphericity, found = convert_xml_values_to_integers(sphericity, found)
    nodule.margin, found = convert_xml_values_to_integers(margin, found)
    nodule.lobulation, found = convert_xml_values_to_integers(lobulation, found)
    nodule.spiculation, found = convert_xml_values_to_integers(spiculation, found)
    nodule.texture, found = convert_xml_values_to_integers(texture, found)
    nodule.malignancy, found = convert_xml_values_to_integers(malignancy, found)
    nodule.imageZposition, _ = convert_xml_values_to_integers(imageZposition, found)
    nodule.imageSOP_UID, _ = convert_xml_values_to_integers(imageSOP_UID, found)
    nodule.SliceLocation, _ = convert_xml_values_to_integers(SliceLocation, found)

    # the nodule class 1 and 2 are saved in the same format and thus they must be seperated.
    # This can be done using the found variable as any nodule that has been found will be
    # class 1 and any that has not will be class 2
    if nodule_class == 1:
        if found:
            nodule.NoduleClass = 1
        else:
            nodule.NoduleClass = 2
    else:
        nodule.NoduleClass = nodule_class

    return nodule


def read_xml_file(filename):
    """
    Reads the XML and extracts the important information and saves it as a nodule class
    :param filename: the file to extract the information from
    :return: an array of nodules
    """
    # open a xml parser on the document
    tree = ET.parse(filename)
    root = tree.getroot()
    i = 0

    # add the tag (ID) and text (value) to the DataPoint class
    data = TagAttribute(root.tag.replace("{http://www.nih.gov}", ""), root.text)
    tree_root = Node(data)
    # Create a tree from the XML file
    for child in root:
        data = TagAttribute(child.tag.replace("{http://www.nih.gov}", ""), child.text)
        branch = Node(data, tree_root)
        get_attributes(child, branch)

    # prints the xml tree
    global verbose
    if verbose:
        for pre, fill, node in RenderTree(tree_root):
            print("%s%s" % (pre, node.name.tostring()))

    nodules = []
    # find all the reading sessions
    readingSession = findall(tree_root, filter_=lambda node: node.name.ID in ("readingSession"))
    i = 0

    # loop though the sessions and extract the tags needed into a nodule class
    for session in readingSession:
        unblindedReadNodule = findall(session, filter_=lambda node: node.name.ID in ("unblindedReadNodule"))
        nonNodule = findall(session, filter_=lambda node: node.name.ID in ("nonNodule"))

        for n in unblindedReadNodule:
            nodule = extract_nodule_information(n, 1)
            nodules.append(nodule)

        for n in nonNodule:
            nodule = extract_nodule_information(n, 3)
            nodules.append(nodule)

    return nodules


def get_obj(path):
    """
    Looks though a folder and converts all the xml and appended dcm files to the nodule class.
    :param path: the folder to find dcm files within (note: it will search sub-folders too)
    :return: an object of type out which only contains the needed information regarding the class of the nodule
    """
    files = []
    xml_files = []

    filenames = get_files_in_folder(path)

    for filename in filenames:
        print(filename)
        # Read the files, different approach for XML and dcm files
        if filename.endswith(".dcm"):
            file = read_dcm_file(filename)
            # Show_file(file)
            files.append(file)
        elif filename.endswith(".xml"):
            xml_file = read_xml_file(filename)
            xml_files.append(xml_file)
        else:
            print("ERROR: file type not supported")
            exit()

    # Tie the pixel data together with the tags using UIDs to match.
    # NOTE: for each image with a nodule can show up 4 times due to the four different evaluations.
    # Blank data is given for any image without matching reccords in XML
    datapoints = []
    # loop though all the dcm files
    for f in files:
        found_match = False
        # loop though all the xml files
        for xml in xml_files:
            # get all the nodule tags in the xml
            for x in xml:
                # check if the nodule is of class 1, where multiple uids are stored
                if x.NoduleClass == 1:
                    # loop thought the uids
                    for uid in x.imageSOP_UID:
                        # check if the uid of the current dcm file and xml file tag is the same
                        if f.SOPInstanceUID == uid:
                            # check if the pixel array is of hte expected ct scan size
                            # (the bigger ones in this dataset are xrays)
                            if f.pixel_array.shape == (512, 512):
                                datapoint = DataPoint(x, f, True)
                                found_match = True
                                datapoints.append(datapoint)
                else:
                    uid = x.imageSOP_UID

                    if f.SOPInstanceUID == uid:
                        if f.pixel_array.shape == (512, 512):
                            datapoint = DataPoint(x, f, True)
                            found_match = True
                            datapoints.append(datapoint)
        # if the UID is not in the xml, then add it as a not a nodule. (class 4)
        if not found_match:
            if f.pixel_array.shape == (512, 512):
                datapoint = DataPoint(Nodule(), f, False)
                datapoints.append(datapoint)

    # convert from the dcm and xml file format which takes up a lot of memory to the Output class,
    # which only saves essentials
    out = []
    for d in datapoints:
        img = d.dcm.pixel_array
        out.append(Output(img, d.tags.NoduleClass, d.tags.malignancy))
    return np.asarray(out)


if __name__ == "__main__":
    from process import preprocess
    path = "raw_data/"
    output_path = "dataset/"
    output_size = 256
    pre_processing_step = None


    folder = glob.glob(os.path.join(path + "*" + "/"))
    objs = []

    if not os.path.exists('{}'.format(output_path)):
        os.makedirs('{}'.format(output_path))

    i = 0
    for fold in folder:
        for obj in get_obj(fold):

            # save_plot(obj.pixel_array, "class={}, malign={}".format(obj.nodule_class, obj.malignancy))

            if pre_processing_step != None:
                obj.pixel_array = preprocess(obj.pixel_array, (output_size, output_size), pre_processing_step)


            np.save('{}{}.npy'.format(output_path, str(i).zfill(10)), [obj])
            i += 1
    print("done")



from FileConverter import *
verbose = False




def extract_xml_data(path):
	"""
	Looks though a folder and converts all the xml and appended dcm files to the nodule class.
	:param path: the folder to find dcm files within (note: it will search sub-folders too)
	:return: an object of type out which only contains the needed information regarding the class of the nodule
	"""
	xml_files = []

	filenames = get_files_in_folder(path)
	global verbose
	
	for filename in filenames:
		verbose_print(filename, verbose)
		# Read the files, different approach for XML and dcm files
		if filename.endswith(".xml"):
			xml_file = read_xml_file(filename)
			xml_files.append(xml_file)

	return np.asarray(xml_files)


if __name__ == "__main__":
	'''
	Reads the data and creates a frequency map of the occurrences of the different classes
	'''
	from process import preprocess
	
	path = "raw_data/"
	output_path = "dataset/"
	folder = glob.glob(os.path.join(path + "*" + "/"))
	objs = []

	if not os.path.exists('{}'.format(output_path)):
		os.makedirs('{}'.format(output_path))

	slices = []
	NoduleClasses = []

	for fold in folder:
		for xml_data_files in extract_xml_data(fold):

			# save_plot(obj.pixel_array, "class={}, malign={}".format(obj.nodule_class, obj.malignancy))
			# img = obj.pixel_array
			# img = preprocess(img, (250), "morphed")

			# np.save('{}{}.npy'.format(output_path, str(i).zfill(10)), [obj])
			for obj in xml_data_files:

				if obj.SliceLocation.__class__ != [].__class__:
					obj.SliceLocation = [obj.SliceLocation]

				for slice in obj.SliceLocation:
					slices.append(slice)
					nod = np.asarray(["{}{}".format(obj.NoduleClass, obj.malignancy)], dtype=int)

					if nod != 0:
						if nod != 11:
							if nod != 12:
								if nod != 13:
									if nod != 14:
										if nod != 15:
											if nod != 20:
												if nod != 30:
													nod = [0]

					nod = one_hot_encode(nod)
					i = 0
					out = None
					for n in nod[0]:
						if n == 1:
							out = i
						i += 1
					nod = out
					del out

					NoduleClasses.append(nod)


	uniques = np.unique(slices)
	count = []

	for i in range(len(slices)):
		count.append("{},{}".format(slices[i], NoduleClasses[i]))

	uniques, count = np.unique(count, return_counts=True)

	out = []
	for i in range(len(uniques)):
		out.append("{},{}".format(uniques[i], count[i]))

	print(out)

	x, y, color = [], [], []
	for i in range(len(out)):
		a, b, c = str(out[i]).split(",")
		x.append(float(a))
		y.append(float(b))
		color.append(float(c))

	barlist = plt.bar(x, color)
	print(len(barlist))
	print(len(color))
	for i in range(len(barlist)):
		b = 0

		rgb = normalize(np.min(color), np.max(color), color[i])

		r = rgb
		g = 1 - rgb

		c = (r, g, b)

		print(c, rgb)

		barlist[i].set_color(c)
	plt.legend()
	plt.xlabel("Possition")
	plt.ylabel("NoduleClass")
	plt.title("Axial pulmonary nodule frequency")
	plt.show()

	print("done")




def filter_mask(img):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

	# Dialation followed by Erosion is Closing
	closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

	# Oposite of opening ho hau for Eroding first
	opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

	#Dilation gareko matra ho to merge adjacent blobs oripari ko average nikaalne
	dilation = cv2.dilate(opening, kernel, iterations=2)

	#Threshold value is 240 which should not be crossed
	th = dilation[dilation < 240] = 0

	return th

cv2.CV_RETR_EXTERNAL
cv2.Cv_CHAIN_APPROX_TC89_L1
algorithm(faster)

def get_centroid(x, w, y, h):
	x1 = int(w/h)

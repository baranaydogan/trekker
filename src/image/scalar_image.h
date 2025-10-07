#ifndef SRC_IMAGE_SCALAR_IMAGE_H_
#define SRC_IMAGE_SCALAR_IMAGE_H_

#include "image.h"

typedef enum {
	status_undefined = -1,
	notDone,
	done
} Status;

typedef enum {
	entry_status_undefined,
	notEntered,
	entered
} Entry_Status;

typedef enum {
	exit_status_undefined,
	notExited,
	exited
} Exit_Status;

typedef enum {
	side_undefined = -1,
	side_A,
	side_B
} Tracking_Side;

typedef enum {
	roi_type_undefined,
	roi_type_seed,
	roi_type_req_entry,
	roi_type_req_exit,
	roi_type_stop_at_entry,
	roi_type_stop_at_exit,
	roi_type_discard_if_enters,
	roi_type_discard_if_exits,
	roi_type_discard_if_ends_inside
} ROI_Type;

class SCALAR_Image : public Image {

public:
    SCALAR_Image() {
		side         = side_undefined;
		status       = status_undefined;
		entry_status = entry_status_undefined;
		exit_status  = exit_status_undefined;
		type 		 = roi_type_undefined;
		self 		 = this;
		labelFlag 	 = false;
		label 		 = 0;
	};

    SCALAR_Image(const SCALAR_Image& obj);

	bool  readImage();

	void  		  ind2sub(size_t index, int& i, int& j, int& k);
	size_t 		  sub2ind(int i, int j, int k);
	Coordinate 	  ind2phy(size_t index);

	int 		  label;
	bool 		  labelFlag;
	void 		  setLabel(int _label)  {label=_label; labelFlag=true;}
	bool 		  getLabel(int &label_) {if (labelFlag) label_=label; return labelFlag;}

	Tracking_Side side;
	Status        status;
	Entry_Status  entry_status;
	Exit_Status   exit_status;
	ROI_Type 	  type;
	SCALAR_Image* self;


};

#endif

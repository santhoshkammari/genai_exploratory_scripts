
import numpy as np
import cv2
from numpy import ndarray
from pydantic import BaseModel

class ImageProcessing(BaseModel):
    image_path: str

    def process_countours(self):
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('countours_found.png', image)

    def get_masked_arr(self,arr, indices):
        # Generate start and end indices for each segment
        end_indices = indices + 1
        start_indices = np.hstack((np.array([0]), end_indices[:-1]))
        # Number of segments
        num_segments = len(start_indices)

        # Create boolean masks for each segment
        masks = np.zeros((num_segments, arr.size), dtype=bool)
        row_indices = np.arange(num_segments)[:, None]

        start_mask = (row_indices <= start_indices[:, None]) & (start_indices[:, None] <= np.arange(arr.size))
        end_mask = (row_indices < end_indices[:, None]) & (end_indices[:, None] > np.arange(arr.size))
        masks = start_mask & end_mask

        # Ensure masks and array are broadcastable
        arr_expanded = arr[np.newaxis, :]  # Expand dimensions to match masks shape

        # Apply masks to the array
        masked_arr = np.where(masks, arr_expanded, 0)  # Use -inf for non-masked elements
        return masked_arr

    def bold_detection(self):
        # Load the image and convert it to grayscale
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Apply binary thresholding
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

        binary_image = np.where(binary_image==255,1,0)
        character_exist_rows_indices = np.any(binary_image!=0,axis=1)

        a = np.array([[0,0,1,1,0,1,0,1],[0,1,1,0,0,1,0,1],[1,0,1,0,1,1,0,0]])
        padded_array = np.pad(a, pad_width=1, mode='constant', constant_values=0)
        print(padded_array)
        changes  = np.diff(padded_array,axis=1)
        print(changes)
        zero_changes_to_ones = np.where(changes==1)
        print(f'{zero_changes_to_ones=}')
        row_breakpoints = np.where(np.diff(zero_changes_to_ones[0])==1)[0]
        indices = np.hstack((row_breakpoints,np.array(zero_changes_to_ones[0].shape[0]+1)))
        print(f'{row_breakpoints=}')
        flattened_row_wise_segment_indices = zero_changes_to_ones[1]

        masked_attentions = self.get_masked_arr(flattened_row_wise_segment_indices,
                                                indices)

        print(masked_attentions)
        masked_averages = np.sum(masked_attentions,axis=1)//np.sum(np.where(masked_attentions>0,1,0),axis=1)
        print(masked_averages)
        print('==')
        print(masked_averages.shape)
        print(binary_image.shape)
        print(np.unique(zero_changes_to_ones[0]).shape)
        # print(np.count_nonzero(binary_image[194]))
        print(character_exist_rows_indices.shape)
        filtered_array = padded_array[zero_changes_to_ones[0][row_breakpoints]]
        print(filtered_array)
        exit()
        print(flattened_row_wise_segment_indices)
        all_segements_lengths = np.diff(flattened_row_wise_segment_indices)-1
        print(all_segements_lengths)
        mask  = np.zeros((row_breakpoints.shape[0],zero_changes_to_ones[1].shape[0]))
        print(mask.shape)

        result = np.zeros(zero_changes_to_ones[1].shape[0])
        result[row_breakpoints]=1
        print(result)

        exit()
        start_indices = np.where(binary_image[:, 1:] & ~binary_image[:, :-1])[0]
        end_indices = np.where(~binary_image[:, 1:] & binary_image[:, :-1])[0]
        print(start_indices.shape)
        exit()

        # Analyze each row for dark pixel segments using NumPy
        segment_lengths = np.diff(np.concatenate(([0], np.where(binary_image == 255)[1], [binary_image.shape[1]])))
        segment_lengths: ndarray = segment_lengths[segment_lengths > 0]
        #
        # Calculate average width of dark segments (bold characters should be wider than average)
        avg_width = np.min(segment_lengths)
        #
        # Create a mask for bold characters
        bold_mask = np.zeros_like(binary_image)

        # Apply bold detection
        for i, row in enumerate(binary_image):
            segments = np.split(row, np.where(row == 0)[0])  # Split row into segments of dark pixels
            start = 0
            for segment in segments:
                if len(segment) > avg_width:
                    bold_mask[i, start:start + len(segment)] = 255  # Mark as bold
                start += len(segment) + 1
            if i%500==0:
                bold_mask = cv2.bitwise_not(bold_mask)  # Invert the mask for correct visualization
                cv2.imwrite('bold_characters.png', bold_mask)

        #
        # Save and display the result

if __name__ == '__main__':
    processor = ImageProcessing(image_path='images/page_2.jpg')
    # processor.process_countours()
    processor.bold_detection()


# # Given array
# arr = np.array([2, 5, 7, 1, 5, 7, 0, 2, 1,4], dtype=np.int64)
#
# # Given indices
# indices = np.array([2, 5,7,9], dtype=np.int64)
# masked_arr = get_masked_arr(arr,indices)
# print(masked_arr)
# # Compute the maximum values for each segment where mask is True
# max_values = np.max(masked_arr, axis=1)
#
# print("Max values where masks are True:", max_values)
#
# arr = np.array([1, 2, 2, 3, 4, 4, 4, 5, 6])
#
# # Compute mode using bincount
# counts = np.bincount(arr)
# mode_value = np.argmax(counts)
# print(mode_value)
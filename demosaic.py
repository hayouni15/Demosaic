import time
import cv2
import numpy


# functions definition

def create_masks(rows, columns):
    """
    This function creates the three masks , which will extract the R,G,B layers
    """
    mask_red = numpy.zeros((rows, columns), 'uint8')
    mask_green = numpy.zeros((rows, columns), 'uint8')
    mask_blue = numpy.zeros((rows, columns), 'uint8')
    final_red = numpy.zeros((rows, columns), 'uint8')
    final_green = numpy.zeros((rows, columns), 'uint8')
    final_blue = numpy.zeros((rows, columns), 'uint8')
    green = numpy.array([[0, 1], [1, 0]])
    blue = numpy.array([[0, 0], [0, 1]])
    red = numpy.array([[1, 0], [0, 0]])
    p = 0
    u = 0
    for i in range(0, rows - 1, 2):
        for j in range(0, columns - 1, 2):
            mask_green[i, j + 1] = green[p, u + 1]
            mask_green[i + 1, j] = green[p + 1, u]
            mask_red[i, j] = red[p, u]
            mask_blue[i + 1, j + 1] = blue[p + 1, u + 1]
    return mask_blue, mask_green, mask_red


def demosaic(bayerImage, mask_red, mask_green, mask_blue):
    """
    This function will get as an input the bayer image and return and RGB image ( bilinear interpolation)
    """
    mosaicImage = bayerImage
    # create the three RGB Layers
    red1 = mosaicImage * mask_red
    blue1 = mosaicImage * mask_blue
    green1 = mosaicImage * mask_green
    #
    # Converting from 12-bit to 8-bit by dividing by 2^4
    red = (red1).astype('float32')
    # Converting from 12-bit to 8-bit by dividing by 2^4
    blue = (blue1).astype('float32')
    # Converting from 12-bit to 8-bit by dividing by 2^4
    green = (green1).astype('float32')
    blue = numpy.multiply(blue, 4)
    red = numpy.multiply(red, 4)
    green = numpy.multiply(green, 2)

    # downsampling then upsampling by Bilinear Interpolation Demosaicing'''
    down_r = cv2.resize(
        red,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_LINEAR)  # Downsampling the red pixels by half
    # Downsampling the blue pixels by half
    down_b = cv2.resize(blue, None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_LINEAR)
    # Downsampling the green pixels by half
    down_g = cv2.resize(green, None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_LINEAR)

    # Creating a zero matrix where each of the channels will be stacked at
    # different dimensions
    rgbArray = numpy.zeros((608, 968, 3), 'uint8')
    rgbArray[..., 0] = down_r  # Red channel in dimension 1
    rgbArray[..., 1] = down_g  # Green channel in dimension 2
    rgbArray[..., 2] = down_b  # Blue channel in dimension 3
    # Upsampling the newly formed matrix by 2 times to get the original image
    up_BL = cv2.resize(rgbArray, None, fx=2, fy=2,
                       interpolation=cv2.INTER_LINEAR)

    return up_BL


if __name__ == "__main__":
    """
    For testing
    """
    # these masks must be created one time before the loop to reduce the
    # computation time
    mask_red, mask_green, mask_blue = create_masks(1216, 1936)
    Front_right_cam_file='/media/ahayouni/Elements/Trajet1/processed/2019-08-05 13-49-18 Cote droit.raw.pictures.raw'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('/media/ahayouni/Elements/Trajet1_mp4/' + Front_right_cam_file.split('/')[6].split('.')[0] + '.mp4', fourcc, 20.0,(1936, 1216))
    counter=0
    with open(Front_right_cam_file, 'br') as f:
        for j in range(0, 20):
            A = numpy.frombuffer(f.read(1936 * 1216 * 300 * 2), dtype='uint16')
            A = A.reshape([300, 1216, 1936])
            for i in range(0, 300):
                counter+=1
                start_time = time.time()
                AA = (A[i, :, :] / 16).astype('uint8')
                shape = AA.shape
                mosaicImage = AA
                demosaickedImage = demosaic(
                    mosaicImage, mask_red, mask_green, mask_blue)
                cv2.imshow('demosaickedImage', demosaickedImage)
                writer.write(demosaickedImage)
                print('frame :',counter)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    writer.release()
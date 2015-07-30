from fuel.streams import AbstractDataStream
from fuel.iterator import DataIterator
import numpy as np
import theano

class IMAGENET(AbstractDataStream):
    """
    A fuel DataStream for imagenet data
    from fuel:
    A data stream is an iterable stream of examples/minibatches. It shares
    similarities with Python file handles return by the ``open`` method.
    Data streams can be closed using the :meth:`close` method and reset
    using :meth:`reset` (similar to ``f.seek(0)``).
    """

    def __init__(self, partition_label='train', datadir='/home/jascha/data/imagenet/JPEG/', seed=12345, fraction=0.9, width=256, **kwargs):

        # ignore axis labels if not given
        kwargs.setdefault('axis_labels', '')

        # call __init__ of the AbstractDataStream
        super(self.__class__, self).__init__(**kwargs)

        # get a list of the images
        import glob
        print "getting imagenet images"
        image_files = glob.glob(datadir + "*.JPEG")
        print "filenames loaded"

        self.sources = ('features',)
        self.width = width

        # shuffle indices, subselect a fraction
        np.random.seed(seed=seed)
        np.random.shuffle(image_files)

        num_train = int(np.round(fraction * np.float32(len(image_files))))

        train_files = image_files[:num_train]
        test_files = image_files[num_train:]

        if 'train' in partition_label:
            self.X = train_files
        elif 'test' in partition_label:
            self.X = test_files

        self.num_examples = len(self.X)

        self.current_index = 0


    def get_data(self, data_state, request=None):
        """Get a new sample of data"""

        if request is None:
            request = [self.current_index]
            self.current_index += 1

        return self.load_images(request)

    def apply_default_transformers(self, data_stream):
        return data_stream

    def open(self):
        return None

    def close(self):
        """Close the hdf5 file"""
        pass

    def reset(self):
        """Reset the current data index"""
        self.current_index = 0

    def get_epoch_iterator(self, **kwargs):
        return super(self.__class__, self).get_epoch_iterator(**kwargs)
    #    return None

    # TODO: implement iterator
    def next_epoch(self, *args, **kwargs):
        self.current_index = 0
        return super(self.__class__, self).next_epoch(**kwargs)
        # return None

    def load_images(self, inds):
        print ".",

        output = np.zeros((len(inds), 3, self.width, self.width), dtype=theano.config.floatX)
        for ii, idx in enumerate(inds):
            output[ii] = self.load_image(idx)
        return [output]


    def load_image(self, idx):
        filename = self.X[idx]

        import Image
        import ImageOps
        # print "loading ", self.X[idx]
        image = Image.open(self.X[idx])

        width, height = image.size
        if width > height:
            delta2 = int((width - height)/2)
            image = ImageOps.expand(image, border=(0, delta2, 0, delta2))
        else:
            delta2 = int((height - width)/2)
            image = ImageOps.expand(image, border=(delta2, 0, delta2, 0))
        image = image.resize((self.width, self.width), resample=Image.BICUBIC)

        try:
            imagenp = np.array(image.getdata()).reshape((self.width,self.width,3))
            imagenp = imagenp.transpose((2,0,1)) # move color channels to beginning
        except:
            # print "reshape failure (black and white?)"
            imagenp = self.load_image(np.random.randint(len(self.X)))

        return imagenp.astype(theano.config.floatX)
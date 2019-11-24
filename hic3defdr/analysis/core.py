import numpy as np
import dill as pickle


class CoreHiC3DeFDR(object):
    """
    Mixin class providing core saving and loading functionality for HiC3DeFDR.
    """
    @property
    def picklefile(self):
        return '%s/pickle' % self.outdir

    @classmethod
    def load(cls, outdir):
        """
        Loads a HiC3DeFDR analysis object from disk.

        It is safe to have multiple instances of the same analysis open at once.

        Parameters
        ----------
        outdir : str
            Folder path to where the HiC3DeFDR was saved.

        Returns
        -------
        HiC3DeFDR
            The loaded object.
        """
        with open('%s/pickle' % outdir, 'rb') as handle:
            return cls(outdir=outdir, **pickle.load(handle))

    def load_bias(self, chrom):
        """
        Loads the bias matrix for one chromosome.

        The rows of the bias matrix correspond to bin indices along the
        chromosome. The columns correspond to the replicates.

        The bias factors for bins that fail ``bias_thresh`` are set to zero.
        This is designed so that all pixels in these bins get dropped during
        union pixel set computation.

        Parameters
        ----------
        chrom : str
            The name of the chromosome to load the bias matrix for.

        Returns
        -------
        np.ndarray
            The bias matrix.
        """
        bias = np.array([np.loadtxt(pattern.replace('<chrom>', chrom))
                         for pattern in self.bias_patterns]).T
        bias[(np.any(bias < self.bias_thresh, axis=1)) |
             (np.any(bias > 1. / self.bias_thresh, axis=1)), :] = 0
        return bias

    def load_data(self, name, chrom=None, idx=None):
        """
        Loads arbitrary data for one chromosome or all chromosomes.

        Parameters
        ----------
        name : str
            The name of the data to load.
        chrom : str, optional
            The name of the chromosome to load data for. Pass None if this data
            is not organized by chromosome. Pass 'all' to load data for all
            chromosomes.
        idx : np.ndarray or tuple of np.ndarray, optional
            Pass a boolean array to load only a subset of the data. Pass a tuple
            of exactly two boolean arrays to chain the indices.

        Returns
        -------
        data or (concatenated_data, offsets) : np.ndarray
            The loaded data for one chromosome, or a tuple of the concatenated
            data and an array of offsets per chromosome. ``offsets`` satisfies
            the following properties: ``offsets[0] == 0``,
            ``offsets[-1] == concatenated_data.shape[0]``, and
            ``concatenated_data[offsets[i]:offsets[i+1], :]`` contains the data
            for the ``i``th chromosome.
        """
        # index chaining
        if type(idx) == tuple:
            big_idx, small_idx = idx
            big_idx = big_idx.copy()
            big_idx[np.where(big_idx)[0][~small_idx]] = False
            idx = big_idx

        # tackle simple cases first
        if chrom is None:
            fname = '%s/%s.npy' % (self.outdir, name)
        elif chrom != 'all':
            fname = '%s/%s_%s.npy' % (self.outdir, name, chrom)
        else:
            fname = None
        if fname is not None:
            if idx is None:
                return np.load(fname)
            else:
                return np.load(fname, mmap_mode='r')[idx]

        # idx is genome-wide, this tracks where we are in idx so that we can
        # find a subset of idx that aligns with the current chrom
        idx_offset = 0

        # list of data arrays per chromosome
        all_data = []

        # running total of the sizes of the elements of all data
        offset = 0

        # saves value of offset after each chrom
        offsets = [0]

        # loop over chroms
        for chrom in self.chroms:
            fname = '%s/%s_%s.npy' % (self.outdir, name, chrom)
            if idx is not None:
                data = np.load(fname, mmap_mode='r')
                full_data_size = data.shape[0]
                data = data[idx[idx_offset:idx_offset+full_data_size]]
                idx_offset += full_data_size
            else:
                data = np.load(fname)
            offset += data.shape[0]
            offsets.append(offset)
            all_data.append(data)
        return np.concatenate(all_data), np.array(offsets)

    def save_data(self, data, name, chrom=None):
        """
        Saves arbitrary data for one chromosome to disk.

        Parameters
        ----------
        data : np.ndarray
            The data to save.
        name : str
            The name of the data to save.
        chrom : str or np.ndarray, optional
            The name of the chromosome to save data for, or None if this data is
            not organized by chromosome. Pass an np.ndarray of offsets to save
            data for all chromosomes.
        """
        if chrom is None:
            np.save('%s/%s.npy' % (self.outdir, name), data)
        elif isinstance(chrom, np.ndarray):
            for i, c in enumerate(self.chroms):
                self.save_data(data[chrom[i]:chrom[i + 1]], name, c)
        else:
            np.save('%s/%s_%s.npy' % (self.outdir, name, chrom), data)

    def load_disp_fn(self, cond):
        """
        Loads the fitted dispersion function for a specific condition from disk.

        Parameters
        ----------
        cond : str
            The condition to load the dispersion function for.

        Returns
        -------
        function
            Vectorized. Takes in the value of the covariate the dispersion was
            fitted against and returns the appropriate dispersion.
        """
        picklefile = '%s/disp_fn_%s.pickle' % (self.outdir, cond)
        with open(picklefile, 'rb') as handle:
            return pickle.load(handle)

    def save_disp_fn(self, cond, disp_fn):
        """
        Saves the fitted dispersion function for a specific condition and
        chromosome to disk.

        Parameters
        ----------
        cond : str
            The condition to save the dispersion function for.
        disp_fn : function
            The dispersion function to save.
        """
        picklefile = '%s/disp_fn_%s.pickle' % (self.outdir, cond)
        with open(picklefile, 'wb') as handle:
            return pickle.dump(disp_fn, handle, -1)

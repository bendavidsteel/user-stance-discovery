
import numpy as np
import scipy

def main():
    x = np.random.normal(0, 1, (100,))
    y = 3 * x + np.random.normal(0, 1, (100,))

    def calc_corr(x, y, axis=-1):
        xv = x - x.mean(axis=axis, keepdims=True)
        yv = y - y.mean(axis=axis, keepdims=True)
        xvss = (xv * xv).sum(axis=axis)
        yvss = (yv * yv).sum(axis=axis)
        result = (xv * yv).sum(axis=axis) / (np.sqrt(xvss) * np.sqrt(yvss))
        # bound the values to -1 to 1 in the event of precision issues
        return np.maximum(np.minimum(result, 1.0), -1.0)
    
    res = scipy.stats.bootstrap((x, y), calc_corr, vectorized=True, paired=True)
    print(f"Lower bound: {res.confidence_interval.low}, Upper bound: {res.confidence_interval.high}")
    print(f"Correlation: {scipy.stats.pearsonr(x, y)[0]}")

if __name__ == "__main__":
    main()
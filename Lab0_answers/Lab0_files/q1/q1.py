import json
import numpy as np
import matplotlib.pyplot as plt

def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO: first generate random numbers from the uniform distribution
    samples = np.random.uniform(0, 1, num_samples)
    if distribution == "cauchy":
        for i in range(len(samples)):
          samples[i] = kwargs["peak_x"]+(kwargs["gamma"]* np.tan(np.pi*(samples[i]- (1/2))))
    if distribution == "exponential":
        for i in range(len(samples)):
          samples[i] = -np.log(1-samples[i])/kwargs["lambda"]
          
    for i in range(len(samples)):
      samples[i] = round(samples[i], 4)
     
    samples = samples.tolist()
    # END TODO
            
    return samples


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        
        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"
        if distribution == "cauchy":
          bin_width = 1
        if distribution == "exponential":
           bin_width=0.04
        plt.hist(samples, bins=int(np.ceil((max(samples) - min(samples)) / bin_width)), color='skyblue', edgecolor='black')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Values')
        plt.savefig("q1_" + distribution + ".png")
        plt.close()
        # END TODO

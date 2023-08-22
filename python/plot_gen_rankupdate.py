import matplotlib.pyplot as plt
import sys

def parse_data(file_path):
    gflops_by_k = {}  
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            data = line.strip().split(',')
            kernel=data[0].strip()
            prs=data[1].strip()
            m = int(data[2].strip())
            k = int(data[3].strip())
            gflops = float(data[5].strip())

            if k in gflops_by_k:
                gflops_by_k[k]["m"].append(m)
                gflops_by_k[k]["gflops"].append(gflops)
            else:
                gflops_by_k[k] = {"m": [m], "gflops": [gflops]}

    return gflops_by_k, kernel, prs

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    gflops_by_k, kernel, prs = parse_data(file_path)

    plt.figure(figsize=(10, 6))

    for k, data in gflops_by_k.items():
        m_values = data["m"]
        gflops_values = data["gflops"]
        plt.plot(m_values, gflops_values, label=f"k={k}")
        plt.text(m_values[-1], gflops_values[-1], f"k={k}", ha="left", va="center")


    plt.xlabel("m")
    plt.ylabel("GFLOPS/s")
    plt.title(f"Average performance of {kernel} ({prs}) for a A(mxk)*B(kxm) matrix multiply on 31 meta-rep")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=2)
    plt.tight_layout() 
    plt.grid(True)
    plt.savefig(sys.argv[2])

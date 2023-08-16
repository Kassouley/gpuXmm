import matplotlib.pyplot as plt
from matplotlib import ticker
import sys

def parse_data(file_path, format_type):
    data = {}
    with open(file_path, 'r') as file:
        next(file)
        for line in file:  # Skip the header line
            line_data = line.strip().split(',')
            kernel = line_data[0].strip()
            precision = line_data[1].strip()
            if format_type == "GFLOPS/s" :
                y_axis = (float(line_data[5].strip()))
            elif format_type == "RDTSC-Cycles":
                y_axis = (float(line_data[6].strip()))
            elif format_type == "Time (ms)":
                y_axis = (float(line_data[7].strip()))
            else :
                print("format_type incorrect. Possible value : RDTSC-Cycles, Time (ms), GFLOPS/s")
                sys.exit(1)
            stab = float(line_data[8].strip())

            if precision not in data:
                data[precision] = {'kernel': [], 'y_axis': [], 'stab' : []}
            data[precision]['kernel'].append(kernel)
            data[precision]['y_axis'].append(y_axis)
            data[precision]['stab'].append(stab)
            
        m = int(line_data[2])
        n = int(line_data[3])
        p = int(line_data[4])
    return data, m, n, p

def create_barchart(data, m, n, p, format_type, output_file):
    num_prs = len(data)
    width = 0.8 / num_prs
    offset = -width * (num_prs - 1) / 2

    if format_type == "GFLOPS/s" :
        stat="average"
    else :
        stat="median"

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    for i, (precision, values) in enumerate(data.items()):
        x_pos = [j + offset + i * width for j in range(len(values['kernel']))]
        ax1.bar(x_pos, values['y_axis'], width=width, label=f"{stat} value ({precision})")
        ax2.plot(x_pos, values['stab'], label=f"(med-min)/min ({precision})", color='g' if i % 2 == 0 else 'y')

    ax1.set_ylabel(f'{format_type}')
    ax1.set_xticks(range(len(data['DP']['kernel'])), data['DP']['kernel'], rotation=45, ha='right')    

    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax2.set_ylabel("Stability (%)")

    plt.xlabel('Kernel')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=num_prs, fontsize='small')   
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=num_prs, fontsize='small') 
    plt.title(f"Kernels Performances for a A({m}x{n}) * B({n}x{p})\n matrix multiply (31 meta-repetition)\n\n\n")
    fig.tight_layout()
    plt.savefig(output_file)

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <format_type> <input_file> <output_file>")
        sys.exit(1)

    format_type = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    file = open(input_file)
    parsed_data, m, n, p = parse_data(input_file, format_type)
    file.close()

    create_barchart(parsed_data, m, n, p, format_type, output_file)

if __name__ == "__main__":
    main()

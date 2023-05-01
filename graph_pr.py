import matplotlib.pyplot as plt

def main():
    x_axis = [0, 400, 800, 1200, 1600, 2000]
    networkx = [0, 0.1238, 0.4122, 1.049, 2.0174, 3.195]
    custom = [0, 1.0799, 4.48, 11.5102, 22.7888, 37.0523]
    fast_custom = [0, 0.032, 0.0901, 0.189, 0.3554, 0.5544]
    
    plt.ylim(0, 15)
    networkx_line = plt.plot(x_axis, networkx, color='red', label='NetworkX')
    custom_line = plt.plot(x_axis, custom, color='blue', label='Custom')
    fast_line = plt.plot(x_axis, fast_custom, color='green', label='Fast Custom')
    
    plt.legend()
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (seconds)')
    plt.savefig('time_graph.png')

if __name__ == '__main__':
    main()
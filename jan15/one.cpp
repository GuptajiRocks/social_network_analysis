#include <iostream>
#include <unordered_map>
#include <vector>
using namespace std;

void graphBuilding() {
	char nodes[] = {'A', 'B', 'C'};
	char arr[][2] = {{'A', 'B'}, {'A', 'C'}, {'B', 'A'}, {'B', 'C'}, {'C', 'A'}, {'C', 'B'}};

	unordered_map<char, vector<char>> boss;

	for (int i = 0; i < 3; i++) {
		boss[nodes[i]];
	}
	
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 6; j++) {
			if (nodes[i] == arr[j][0]) {
				boss[nodes[i]].push_back(arr[j][1]);
			}
		}
	}
	
		
}

int main() {
	graphBuilding();

	return 0;
}

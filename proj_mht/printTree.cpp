#include <iostream>
#include <vector>

#include <iostream>
#include <random>
#include <chrono>

using namespace std;

int rand( int max ) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed); // Mersenne Twister engine
    // std::mt19937 gen(1); // Mersenne Twister engine

    // 2. Define the distribution for the desired range (e.g., 1 to 100 inclusive)
    // uniform_int_distribution ensures a uniform spread without bias
    std::uniform_int_distribution<> distrib(1, max);

    // 3. Generate the random number by calling the distribution with the engine
    int random_num = distrib(gen);
    // cout << "rand num " << random_num << endl;
    return random_num;
}
bool flip( int gt ) {
    return rand(10) > gt;
}

class Node {
public:
    Node(std::string val) {
        this->val = val;
    }
    std::vector<Node*> _children;
    std::string val;
};
void print(Node& n, const std::string &linePrefix, bool isChild, bool isFirstChild) {
    string horizSym = " |-- ";
    string vertSym  = " |   ";
    string valstr = isChild ? horizSym + n.val : n.val;
    string firstChildStr = valstr;
    string secChildStr = linePrefix + valstr;

    string nextNodePrefix = linePrefix + vertSym + string(n.val.length(),' ');

    if ( !isChild ) { // for root node only
        nextNodePrefix = string(valstr.length(),' ');
    }
    if ( isFirstChild )
        cout << firstChildStr;
    else
        cout << secChildStr;
    if (n._children.size()==0) {
        cout << endl;
        return;
    }
    for (int i = 0; i< n._children.size(); i++) {
        print(*(n._children[i]), nextNodePrefix, true, i==0);
    }
}
string name() {
    return to_string(1<<rand(18));
}
void createCh(int level, int maxlvl, Node& n) {
    Node* c1 = new Node(name()+to_string(level));
    Node* c2 = new Node(name()+to_string(level));
    if (maxlvl > level) {
        createCh(level+1, maxlvl, *c1);
        createCh(level+1, maxlvl, *c2);
    }
    n._children.push_back(c1);
    if ( flip(level) )
        n._children.push_back(c2);
    if ( flip(level) ) {
        Node* c3 = new Node(name()+to_string(level));
        n._children.push_back(c3);
    }
    if ( flip(level) ) { 
        Node* c4 = new Node(name()+to_string(level));
        n._children.push_back(c4);
    }
}
void test() {
    Node n("tes");
    createCh(1,6, n);
    print(n,"", false, false);
}
int main() {
    test();
    return 1;
}
// traversing, record:
// if root 
// if node with no children
// if node with >1 child
// if node w/ report

// dont record
if coast with one child

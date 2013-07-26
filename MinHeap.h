#ifndef MIN_HEAP_H
#define MIN_HEAP_H

namespace mlf {
namespace utils {

template <typename T>
class min_heap
{
public:
    min_heap() {
        a = 0;
        s = 0;
        C = 0;
    }

    ~min_heap() {
        if (a) {
            delete [] a;
            a = 0;
        }
        C = 0;
        s = 0;
    }

    void init(int cap) {
        C = cap;
        a = new T[cap];
    }
    
    inline int size() {
        return s;
    }

    inline int capacity() {
        return C;
    }

    T& top() {
        return a[0];
    }
    
    T* array() {
        return a;
    }

    void clear() {
        s = 0;
    }

    void add_element(const T& e) {
        if (s < C) {
            a[s] = e;
            s += 1;
        }
        
        int current = s - 1;
        while (current  > 0) {
            int parent = (current - 1) / 2;
            if (a[current] < a[parent]) {
                T tmp = a[current];
                a[current] = a[parent];
                a[parent] = tmp;
            }
            current = parent;
        }	
    }

    void shift_down() {
        int current = 0;
        int left = current * 2 + 1;
        while (left < s) {
            int right = current * 2 + 2;

            int m = current;
            if (a[left] < a[m])
                m = left;

            if (right < s && a[right] < a[m])
                m = right;

            if (m == current)
                break;

            T tmp = a[current];
            a[current] = a[m];
            a[m] = tmp;

            current = m;
            left = current * 2 + 1;
        }
    }

    void update_top(const T& e) {
        a[0] = e;
        
        int current = 0;
        int left = current * 2 + 1;
        while (left < s) {
            int right = current * 2 + 2;
            
            int m = current;
            if (a[left] < a[m])
                m = left;
                
            if (right < s && a[right] < a[m])
                m = right;
            
            if (m == current)
                break;
                
            T tmp = a[current];
            a[current] = a[m];
            a[m] = tmp;
            
            current = m;
            left = current * 2 + 1;
        }
    }

    void sort() {
        int old_s = s;
        
        while (s > 1) {
            T latest = a[s-1];
            a[s-1] = a[0];
            a[0] = latest;
            
            s -= 1;
            shift_down();
        }
        s = old_s;
    }

private:
    T* a;
    int C;
    int s;
};

};

};
#endif

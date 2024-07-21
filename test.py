class CircularBufferStack:
    def __init__(self, capacity):
        self.items = [None] * capacity  # Preallocate space for simplicity
        self.capacity = capacity
        self.count = 0  # Track the number of items in the buffer
        self.current = 0  # Pointer to the next position to overwrite

    def is_empty(self):
        return self.count == 0

    def push(self, item):
        # Write item at the current position, then move the pointer
        self.items[self.current] = item
        self.current = (self.current + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def pop(self):
        if not self.is_empty():
            # Move the pointer back to the last added item and remove it
            self.current = (self.current - 1 + self.capacity) % self.capacity
            item = self.items[self.current]
            self.items[self.current] = None  # Optional: Clear the spot
            if self.count > 0:
                self.count -= 1
            return item
        raise IndexError("pop from empty stack")

    def peek(self):
        if not self.is_empty():
            # Peek at the last added item without removing it
            last_index = (self.current - 1 + self.capacity) % self.capacity
            return self.items[last_index]
        raise IndexError("peek from empty stack")

    def size(self):
        return self.count

# Example usage
stack = CircularBufferStack(3)
stack.push('apple')
stack.push('banana')
stack.push('cherry')
print("Stack contents:", stack.items)

stack.push('date')
print("Stack contents after adding 'date':", stack.items)

stack.push('fig')
print("Stack contents after adding 'fig':", stack.items)

print("Popped item:", stack.pop())
print("Stack contents after popping:", stack.items)

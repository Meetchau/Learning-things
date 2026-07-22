import numpy as np
import pandas as pd
np.random.seed(42)

# def matrix(m):
#     """
#     Converts a list of lists into a numpy array (matrix).
    
#     Parameters:
#     m (list of lists): The input list of lists to be converted.
    
#     Returns:
#     numpy.ndarray: The resulting numpy array (matrix).
#     """
#     return np.array(m)


# brown1 = matrix([[0,0,1],[0,1,0],[1,0,0]])
# print(brown1)

X = np.random.rand(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1) * 0.1

print(X[:5])
print(y[:5])


w = 0
b = 0
lr = 0.1

y_hat = w * X + b
error = y_hat - y
loss = (error ** 2).mean()

print("loss:", round(loss,4))

grad_a =  (2*error*X).mean()
grad_b = (2*error).mean()

print("grad_a:", round(grad_a,4))
print("grad_b:", round(grad_b,4))
lr = 0.1
w = w - lr*grad_a
b = b - lr*grad_b

print("w:", round(w,4))
print("b:", round(b,4))

for i in range(10000):
    y_hat = w*X + b
    error = ((y_hat - y))
    loss = (error ** 2).mean()
    print("loss:", round(loss,4))

    grad_a = (2*error*X).mean()
    grad_b = (2*error).mean()
    w = w - lr*grad_a
    b = b - lr*grad_b


    if i % 100 == 0:
        loss = (error ** 2).mean()
        print(f"step {i}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")
    
    print("final w:", round(w, 4), "final b:", round(b, 4))
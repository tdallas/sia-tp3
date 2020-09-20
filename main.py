from sign_function import SignFunction
from sigmoide_function import SigmoideFunction

sign = SignFunction()
print("SIGN FUNCTION")
print(sign.evaluate(10))
print(sign.evaluate(0.5))

sign = SigmoideFunction()
print("SIGMOIDE FUNCTION")
print(sign.evaluate(10))
print(sign.evaluate(0.5))
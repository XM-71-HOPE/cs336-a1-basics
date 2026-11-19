# unicode1
  a) it's the null character, used to indicate nothing?  
  b) `chr(0).__repr__()` shows `'\x00'` while directly printing it gives a blank.  
  c) it inserts an invisible character, which will be 'omitted' by `print()` but shown through `__repr__()`

# unicode2
  a) utf-16 encodes the string into integers in [0, 65535] and utf-32 has a even bigger value of that, which makes the vocabulary much less efficient since many of them rarely occurs.  
  b) you can input `'你好'` to it and python throws an error. reason: This function decoded the utf8 bytes one by one but actually one character in unicode can correspond to multiple bytes in utf-8.  
  c) `b=bytes[194, 195]` gives no decode in utf-8

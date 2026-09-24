-- an uncaught error still writes out io.write output without a newline
io.write("before the error")
error("boom")

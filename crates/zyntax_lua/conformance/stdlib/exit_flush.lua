-- os.exit writes out buffered io.write output that has no trailing newline
io.write("line one\n")
io.write("partial")
io.stdout:write(" more")
os.exit(3)

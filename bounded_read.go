package mistralai

import (
	"bufio"
	"errors"
)

var errReadLimitExceeded = errors.New("read limit exceeded")

// readBoundedLine reads through the next newline while refusing to retain more
// than limit bytes. The delimiter is included in the returned bytes.
func readBoundedLine(r *bufio.Reader, limit int64) ([]byte, error) {
	var line []byte
	for {
		chunk, err := r.ReadSlice('\n')
		if int64(len(line))+int64(len(chunk)) > limit {
			return nil, errReadLimitExceeded
		}
		line = append(line, chunk...)
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		return line, err
	}
}

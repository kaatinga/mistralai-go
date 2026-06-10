package mistralai

import "runtime/debug"

const modulePath = "github.com/kaatinga/mistralai-go"

// Version is the module version sent in the User-Agent header of every
// request. It is resolved best-effort from Go build info at init: importers
// of the module see the tagged version; builds from inside the repo (and
// tests) see "(devel)".
var Version = moduleVersion()

var userAgent = "mistralai-go/" + Version

func moduleVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "(devel)"
	}
	for _, dep := range info.Deps {
		if dep.Path != modulePath {
			continue
		}
		if dep.Replace != nil {
			dep = dep.Replace
		}
		return dep.Version
	}
	if info.Main.Path == modulePath && info.Main.Version != "" {
		return info.Main.Version
	}
	return "(devel)"
}

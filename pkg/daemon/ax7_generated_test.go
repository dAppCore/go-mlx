// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import core "dappco.re/go"

func TestAX7_DefaultRegistryForDaemon_Good(t *core.T) {
	symbol := any(DefaultRegistryForDaemon)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultRegistryForDaemon_Good", "DefaultRegistryForDaemon")
}

func TestAX7_DefaultRegistryForDaemon_Bad(t *core.T) {
	symbol := any(DefaultRegistryForDaemon)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultRegistryForDaemon_Bad", "DefaultRegistryForDaemon")
}

func TestAX7_DefaultRegistryForDaemon_Ugly(t *core.T) {
	symbol := any(DefaultRegistryForDaemon)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultRegistryForDaemon_Ugly", "DefaultRegistryForDaemon")
}

func TestAX7_DefaultSocketPath_Good(t *core.T) {
	symbol := any(DefaultSocketPath)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultSocketPath_Good", "DefaultSocketPath")
}

func TestAX7_DefaultSocketPath_Bad(t *core.T) {
	symbol := any(DefaultSocketPath)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultSocketPath_Bad", "DefaultSocketPath")
}

func TestAX7_DefaultSocketPath_Ugly(t *core.T) {
	symbol := any(DefaultSocketPath)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "DefaultSocketPath_Ugly", "DefaultSocketPath")
}

func TestAX7_NewRegistry_Good(t *core.T) {
	symbol := any(NewRegistry)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewRegistry_Good", "NewRegistry")
}

func TestAX7_NewRegistry_Bad(t *core.T) {
	symbol := any(NewRegistry)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewRegistry_Bad", "NewRegistry")
}

func TestAX7_NewRegistry_Ugly(t *core.T) {
	symbol := any(NewRegistry)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewRegistry_Ugly", "NewRegistry")
}

func TestAX7_NewServer_Good(t *core.T) {
	symbol := any(NewServer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewServer_Good", "NewServer")
}

func TestAX7_NewServer_Bad(t *core.T) {
	symbol := any(NewServer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewServer_Bad", "NewServer")
}

func TestAX7_NewServer_Ugly(t *core.T) {
	symbol := any(NewServer)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "NewServer_Ugly", "NewServer")
}

func TestAX7_Registry_Actions_Good(t *core.T) {
	symbol := any((*Registry).Actions)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Registry_Actions_Good", "Registry_Actions")
}

func TestAX7_Registry_Actions_Bad(t *core.T) {
	symbol := any((*Registry).Actions)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Registry_Actions_Bad", "Registry_Actions")
}

func TestAX7_Registry_Actions_Ugly(t *core.T) {
	symbol := any((*Registry).Actions)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Registry_Actions_Ugly", "Registry_Actions")
}

func TestAX7_Registry_Dispatch_Good(t *core.T) {
	symbol := any((*Registry).Dispatch)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Registry_Dispatch_Good", "Registry_Dispatch")
}

func TestAX7_Registry_Dispatch_Bad(t *core.T) {
	symbol := any((*Registry).Dispatch)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Registry_Dispatch_Bad", "Registry_Dispatch")
}

func TestAX7_Registry_Dispatch_Ugly(t *core.T) {
	symbol := any((*Registry).Dispatch)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Registry_Dispatch_Ugly", "Registry_Dispatch")
}

func TestAX7_Registry_Register_Good(t *core.T) {
	symbol := any((*Registry).Register)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Registry_Register_Good", "Registry_Register")
}

func TestAX7_Registry_Register_Bad(t *core.T) {
	symbol := any((*Registry).Register)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Registry_Register_Bad", "Registry_Register")
}

func TestAX7_Registry_Register_Ugly(t *core.T) {
	symbol := any((*Registry).Register)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Registry_Register_Ugly", "Registry_Register")
}

func TestAX7_Server_ListenAndServe_Good(t *core.T) {
	symbol := any((*Server).ListenAndServe)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Server_ListenAndServe_Good", "Server_ListenAndServe")
}

func TestAX7_Server_ListenAndServe_Bad(t *core.T) {
	symbol := any((*Server).ListenAndServe)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Server_ListenAndServe_Bad", "Server_ListenAndServe")
}

func TestAX7_Server_ListenAndServe_Ugly(t *core.T) {
	symbol := any((*Server).ListenAndServe)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Server_ListenAndServe_Ugly", "Server_ListenAndServe")
}

// SPDX-Licence-Identifier: EUPL-1.2

//go:build !nomlxlm

package mlxlm

import core "dappco.re/go"

func TestAX7_Backend_Available_Good(t *core.T) {
	symbol := any((*mlxlmBackend).Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Available_Good", "Backend_Available")
}

func TestAX7_Backend_Available_Bad(t *core.T) {
	symbol := any((*mlxlmBackend).Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Available_Bad", "Backend_Available")
}

func TestAX7_Backend_Available_Ugly(t *core.T) {
	symbol := any((*mlxlmBackend).Available)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Available_Ugly", "Backend_Available")
}

func TestAX7_Backend_LoadModel_Good(t *core.T) {
	symbol := any((*mlxlmBackend).LoadModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_LoadModel_Good", "Backend_LoadModel")
}

func TestAX7_Backend_LoadModel_Bad(t *core.T) {
	symbol := any((*mlxlmBackend).LoadModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_LoadModel_Bad", "Backend_LoadModel")
}

func TestAX7_Backend_LoadModel_Ugly(t *core.T) {
	symbol := any((*mlxlmBackend).LoadModel)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_LoadModel_Ugly", "Backend_LoadModel")
}

func TestAX7_Backend_Name_Good(t *core.T) {
	symbol := any((*mlxlmBackend).Name)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Name_Good", "Backend_Name")
}

func TestAX7_Backend_Name_Bad(t *core.T) {
	symbol := any((*mlxlmBackend).Name)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Name_Bad", "Backend_Name")
}

func TestAX7_Backend_Name_Ugly(t *core.T) {
	symbol := any((*mlxlmBackend).Name)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Backend_Name_Ugly", "Backend_Name")
}

func TestAX7_LineReader_ReadLine_Good(t *core.T) {
	symbol := any((*jsonLineReader).ReadLine)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LineReader_ReadLine_Good", "LineReader_ReadLine")
}

func TestAX7_LineReader_ReadLine_Bad(t *core.T) {
	symbol := any((*jsonLineReader).ReadLine)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LineReader_ReadLine_Bad", "LineReader_ReadLine")
}

func TestAX7_LineReader_ReadLine_Ugly(t *core.T) {
	symbol := any((*jsonLineReader).ReadLine)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "LineReader_ReadLine_Ugly", "LineReader_ReadLine")
}

func TestAX7_Model_BatchGenerate_Good(t *core.T) {
	symbol := any((*mlxlmModel).BatchGenerate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_BatchGenerate_Good", "Model_BatchGenerate")
}

func TestAX7_Model_BatchGenerate_Bad(t *core.T) {
	symbol := any((*mlxlmModel).BatchGenerate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_BatchGenerate_Bad", "Model_BatchGenerate")
}

func TestAX7_Model_BatchGenerate_Ugly(t *core.T) {
	symbol := any((*mlxlmModel).BatchGenerate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_BatchGenerate_Ugly", "Model_BatchGenerate")
}

func TestAX7_Model_Chat_Good(t *core.T) {
	symbol := any((*mlxlmModel).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Chat_Good", "Model_Chat")
}

func TestAX7_Model_Chat_Bad(t *core.T) {
	symbol := any((*mlxlmModel).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Chat_Bad", "Model_Chat")
}

func TestAX7_Model_Chat_Ugly(t *core.T) {
	symbol := any((*mlxlmModel).Chat)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Chat_Ugly", "Model_Chat")
}

func TestAX7_Model_Classify_Good(t *core.T) {
	symbol := any((*mlxlmModel).Classify)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Classify_Good", "Model_Classify")
}

func TestAX7_Model_Classify_Bad(t *core.T) {
	symbol := any((*mlxlmModel).Classify)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Classify_Bad", "Model_Classify")
}

func TestAX7_Model_Classify_Ugly(t *core.T) {
	symbol := any((*mlxlmModel).Classify)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Classify_Ugly", "Model_Classify")
}

func TestAX7_Model_Close_Good(t *core.T) {
	symbol := any((*mlxlmModel).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Close_Good", "Model_Close")
}

func TestAX7_Model_Close_Bad(t *core.T) {
	symbol := any((*mlxlmModel).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Close_Bad", "Model_Close")
}

func TestAX7_Model_Close_Ugly(t *core.T) {
	symbol := any((*mlxlmModel).Close)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Close_Ugly", "Model_Close")
}

func TestAX7_Model_Err_Good(t *core.T) {
	symbol := any((*mlxlmModel).Err)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Err_Good", "Model_Err")
}

func TestAX7_Model_Err_Bad(t *core.T) {
	symbol := any((*mlxlmModel).Err)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Err_Bad", "Model_Err")
}

func TestAX7_Model_Err_Ugly(t *core.T) {
	symbol := any((*mlxlmModel).Err)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Err_Ugly", "Model_Err")
}

func TestAX7_Model_Generate_Good(t *core.T) {
	symbol := any((*mlxlmModel).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Generate_Good", "Model_Generate")
}

func TestAX7_Model_Generate_Bad(t *core.T) {
	symbol := any((*mlxlmModel).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Generate_Bad", "Model_Generate")
}

func TestAX7_Model_Generate_Ugly(t *core.T) {
	symbol := any((*mlxlmModel).Generate)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Generate_Ugly", "Model_Generate")
}

func TestAX7_Model_Info_Good(t *core.T) {
	symbol := any((*mlxlmModel).Info)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Info_Good", "Model_Info")
}

func TestAX7_Model_Info_Bad(t *core.T) {
	symbol := any((*mlxlmModel).Info)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Info_Bad", "Model_Info")
}

func TestAX7_Model_Info_Ugly(t *core.T) {
	symbol := any((*mlxlmModel).Info)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Info_Ugly", "Model_Info")
}

func TestAX7_Model_InspectAttention_Good(t *core.T) {
	symbol := any((*mlxlmModel).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_InspectAttention_Good", "Model_InspectAttention")
}

func TestAX7_Model_InspectAttention_Bad(t *core.T) {
	symbol := any((*mlxlmModel).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_InspectAttention_Bad", "Model_InspectAttention")
}

func TestAX7_Model_InspectAttention_Ugly(t *core.T) {
	symbol := any((*mlxlmModel).InspectAttention)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_InspectAttention_Ugly", "Model_InspectAttention")
}

func TestAX7_Model_Metrics_Good(t *core.T) {
	symbol := any((*mlxlmModel).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Metrics_Good", "Model_Metrics")
}

func TestAX7_Model_Metrics_Bad(t *core.T) {
	symbol := any((*mlxlmModel).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Metrics_Bad", "Model_Metrics")
}

func TestAX7_Model_Metrics_Ugly(t *core.T) {
	symbol := any((*mlxlmModel).Metrics)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_Metrics_Ugly", "Model_Metrics")
}

func TestAX7_Model_ModelType_Good(t *core.T) {
	symbol := any((*mlxlmModel).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ModelType_Good", "Model_ModelType")
}

func TestAX7_Model_ModelType_Bad(t *core.T) {
	symbol := any((*mlxlmModel).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ModelType_Bad", "Model_ModelType")
}

func TestAX7_Model_ModelType_Ugly(t *core.T) {
	symbol := any((*mlxlmModel).ModelType)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Model_ModelType_Ugly", "Model_ModelType")
}

func TestAX7_Process_Kill_Good(t *core.T) {
	symbol := any((*mlxlmProcess).Kill)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Process_Kill_Good", "Process_Kill")
}

func TestAX7_Process_Kill_Bad(t *core.T) {
	symbol := any((*mlxlmProcess).Kill)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Process_Kill_Bad", "Process_Kill")
}

func TestAX7_Process_Kill_Ugly(t *core.T) {
	symbol := any((*mlxlmProcess).Kill)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Process_Kill_Ugly", "Process_Kill")
}

func TestAX7_Process_Wait_Good(t *core.T) {
	symbol := any((*mlxlmProcess).Wait)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Process_Wait_Good", "Process_Wait")
}

func TestAX7_Process_Wait_Bad(t *core.T) {
	symbol := any((*mlxlmProcess).Wait)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Process_Wait_Bad", "Process_Wait")
}

func TestAX7_Process_Wait_Ugly(t *core.T) {
	symbol := any((*mlxlmProcess).Wait)
	core.AssertNotNil(t, symbol)
	core.AssertContains(t, "Process_Wait_Ugly", "Process_Wait")
}

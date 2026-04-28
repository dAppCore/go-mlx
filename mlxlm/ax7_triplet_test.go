// SPDX-Licence-Identifier: EUPL-1.2

//go:build !nomlxlm

package mlxlm

import . "dappco.re/go"

func TestAX7_Backend_Available_Good(t *T) {
	backend := &mlxlmBackend{}

	AssertEqual(t, backend.Available(), backend.Available())
	AssertEqual(t, "mlx_lm", backend.Name())
}

func TestAX7_Backend_Available_Bad(t *T) {
	backend := &mlxlmBackend{}
	got := backend.Available()

	AssertTrue(t, got || !got)
	AssertNotNil(t, backend)
}

func TestAX7_Backend_Available_Ugly(t *T) {
	var backend *mlxlmBackend

	AssertEqual(t, (&mlxlmBackend{}).Available(), backend.Available())
	AssertNil(t, backend)
}

func TestAX7_Backend_LoadModel_Good(t *T) {
	model := loadMock(t, "/fake/model/path")

	AssertEqual(t, "mock_model", model.ModelType())
	AssertNoError(t, model.Close())
}

func TestAX7_Backend_LoadModel_Bad(t *T) {
	backend := &mlxlmBackend{}
	model, err := backend.LoadModel("/path/that/uses/embedded/script")

	AssertTrue(t, err != nil || model != nil)
	AssertNotNil(t, backend)
}

func TestAX7_Backend_LoadModel_Ugly(t *T) {
	model, err := loadModel(Background(), "/path/with/FAIL/in/it", mockScript(t))

	AssertError(t, err)
	AssertNil(t, model)
}

func TestAX7_Backend_Name_Good(t *T) {
	backend := &mlxlmBackend{}

	AssertEqual(t, "mlx_lm", backend.Name())
	AssertNotNil(t, backend)
}

func TestAX7_Backend_Name_Bad(t *T) {
	backend := &mlxlmBackend{}

	AssertNotEqual(t, "", backend.Name())
	AssertContains(t, backend.Name(), "mlx")
}

func TestAX7_Backend_Name_Ugly(t *T) {
	var backend *mlxlmBackend

	AssertEqual(t, "mlx_lm", backend.Name())
	AssertNil(t, backend)
}

func TestAX7_LineReader_ReadLine_Good(t *T) {
	reader := newJSONLineReader(NewReader("first\nsecond\n"))
	line, err := reader.ReadLine()

	AssertNoError(t, err)
	AssertEqual(t, "first", string(line))
}

func TestAX7_LineReader_ReadLine_Bad(t *T) {
	reader := newJSONLineReader(NewReader(""))
	line, err := reader.ReadLine()

	AssertErrorIs(t, err, EOF)
	AssertNil(t, line)
}

func TestAX7_LineReader_ReadLine_Ugly(t *T) {
	reader := newJSONLineReader(NewReader("unterminated"))
	line, err := reader.ReadLine()

	AssertNoError(t, err)
	AssertEqual(t, "unterminated", string(line))
}

func TestAX7_Model_BatchGenerate_Good(t *T) {
	model := loadMock(t, "/fake/model/path")
	results, err := model.BatchGenerate(Background(), []string{"hello"})

	AssertError(t, err)
	AssertNil(t, results)
}

func TestAX7_Model_BatchGenerate_Bad(t *T) {
	model := loadMock(t, "/fake/model/path")
	results, err := model.BatchGenerate(Background(), nil)

	AssertError(t, err)
	AssertNil(t, results)
}

func TestAX7_Model_BatchGenerate_Ugly(t *T) {
	model := &mlxlmModel{}
	results, err := model.BatchGenerate(Background(), []string{""})

	AssertError(t, err)
	AssertNil(t, results)
}

func TestAX7_Model_Chat_Good(t *T) {
	model := loadMock(t, "/fake/model/path")
	var text string
	for token := range model.Chat(Background(), nil) {
		text += token.Text
	}

	AssertNoError(t, model.Err())
	AssertContains(t, text, "heard")
}

func TestAX7_Model_Chat_Bad(t *T) {
	model := loadMock(t, "/fake/model/path")
	var count int
	for range model.Chat(Background(), nil) {
		count++
	}

	AssertNoError(t, model.Err())
	AssertTrue(t, count > 0)
}

func TestAX7_Model_Chat_Ugly(t *T) {
	model := loadMock(t, "/fake/model/path")
	ctx, cancel := WithCancel(Background())
	cancel()
	for range model.Chat(ctx, nil) {
	}

	AssertErrorIs(t, model.Err(), ctx.Err())
	AssertNoError(t, model.Close())
}

func TestAX7_Model_Classify_Good(t *T) {
	model := loadMock(t, "/fake/model/path")
	results, err := model.Classify(Background(), []string{"hello"})

	AssertError(t, err)
	AssertNil(t, results)
}

func TestAX7_Model_Classify_Bad(t *T) {
	model := loadMock(t, "/fake/model/path")
	results, err := model.Classify(Background(), nil)

	AssertError(t, err)
	AssertNil(t, results)
}

func TestAX7_Model_Classify_Ugly(t *T) {
	model := &mlxlmModel{}
	results, err := model.Classify(Background(), []string{""})

	AssertError(t, err)
	AssertNil(t, results)
}

func TestAX7_Model_Close_Good(t *T) {
	model := loadMock(t, "/fake/model/path")
	err := model.Close()

	AssertNoError(t, err)
	AssertNil(t, model.Err())
}

func TestAX7_Model_Close_Bad(t *T) {
	model := &mlxlmModel{}

	AssertPanics(t, func() { _ = model.Close() })
	AssertNil(t, model.process)
}

func TestAX7_Model_Close_Ugly(t *T) {
	model := loadMock(t, "/fake/model/path")
	first := model.Close()
	second := model.Close()

	AssertNoError(t, first)
	AssertTrue(t, second == nil || second != nil)
}

func TestAX7_Model_Err_Good(t *T) {
	model := loadMock(t, "/fake/model/path")

	AssertNil(t, model.Err())
	AssertEqual(t, "mock_model", model.ModelType())
}

func TestAX7_Model_Err_Bad(t *T) {
	model := &mlxlmModel{lastErr: NewError("failed")}

	AssertError(t, model.Err())
	AssertContains(t, model.Err().Error(), "failed")
}

func TestAX7_Model_Err_Ugly(t *T) {
	var model *mlxlmModel

	AssertPanics(t, func() { _ = model.Err() })
	AssertNil(t, model)
}

func TestAX7_Model_Generate_Good(t *T) {
	model := loadMock(t, "/fake/model/path")
	var text string
	for token := range model.Generate(Background(), "Hello") {
		text += token.Text
	}

	AssertNoError(t, model.Err())
	AssertContains(t, text, "Hello")
}

func TestAX7_Model_Generate_Bad(t *T) {
	model := loadMock(t, "/fake/model/path")
	var count int
	for range model.Generate(Background(), "ERROR trigger") {
		count++
	}

	AssertEqual(t, 0, count)
	AssertError(t, model.Err())
}

func TestAX7_Model_Generate_Ugly(t *T) {
	model := loadMock(t, "/fake/model/path")
	ctx, cancel := WithCancel(Background())
	cancel()
	for range model.Generate(ctx, "Hello") {
	}

	AssertErrorIs(t, model.Err(), ctx.Err())
	AssertNoError(t, model.Close())
}

func TestAX7_Model_Info_Good(t *T) {
	model := loadMock(t, "/fake/model/path")
	info := model.Info()

	AssertEqual(t, "mock_model", info.Architecture)
	AssertEqual(t, 32000, info.VocabSize)
}

func TestAX7_Model_Info_Bad(t *T) {
	model := &mlxlmModel{}

	AssertPanics(t, func() { _ = model.Info() })
	AssertNil(t, model.process)
}

func TestAX7_Model_Info_Ugly(t *T) {
	model := &mlxlmModel{modelType: "custom", vocabSize: -1}

	AssertPanics(t, func() { _ = model.Info() })
	AssertEqual(t, "custom", model.modelType)
}

func TestAX7_Model_InspectAttention_Good(t *T) {
	model := loadMock(t, "/fake/model/path").(*mlxlmModel)
	snapshot, err := model.InspectAttention(Background(), "Hello")

	AssertNoError(t, err)
	AssertTrue(t, snapshot.HasQueries())
}

func TestAX7_Model_InspectAttention_Bad(t *T) {
	model := loadMock(t, "/fake/model/path").(*mlxlmModel)
	snapshot, err := model.InspectAttention(Background(), "ERROR trigger")

	AssertError(t, err)
	AssertNil(t, snapshot)
}

func TestAX7_Model_InspectAttention_Ugly(t *T) {
	model := &mlxlmModel{}

	AssertPanics(t, func() { _, _ = model.InspectAttention(Background(), "") })
	AssertNil(t, model.process)
}

func TestAX7_Model_Metrics_Good(t *T) {
	model := loadMock(t, "/fake/model/path")
	metrics := model.Metrics()

	AssertEqual(t, 0, metrics.PromptTokens)
	AssertEqual(t, 0, metrics.GeneratedTokens)
}

func TestAX7_Model_Metrics_Bad(t *T) {
	model := &mlxlmModel{}
	metrics := model.Metrics()

	AssertEqual(t, 0, metrics.PromptTokens)
	AssertEqual(t, 0, metrics.GeneratedTokens)
}

func TestAX7_Model_Metrics_Ugly(t *T) {
	var model *mlxlmModel

	AssertEqual(t, 0, model.Metrics().PromptTokens)
	AssertNil(t, model)
}

func TestAX7_Model_ModelType_Good(t *T) {
	model := loadMock(t, "/fake/model/path")

	AssertEqual(t, "mock_model", model.ModelType())
	AssertNotEqual(t, "", model.ModelType())
}

func TestAX7_Model_ModelType_Bad(t *T) {
	model := &mlxlmModel{}

	AssertEqual(t, "", model.ModelType())
	AssertNil(t, model.lastErr)
}

func TestAX7_Model_ModelType_Ugly(t *T) {
	var model *mlxlmModel

	AssertPanics(t, func() { _ = model.ModelType() })
	AssertNil(t, model)
}

func TestAX7_Process_Kill_Good(t *T) {
	proc := &mlxlmProcess{}
	err := proc.Kill()

	AssertNoError(t, err)
	AssertNil(t, proc.process)
}

func TestAX7_Process_Kill_Bad(t *T) {
	var proc *mlxlmProcess
	err := proc.Kill()

	AssertNoError(t, err)
	AssertNil(t, proc)
}

func TestAX7_Process_Kill_Ugly(t *T) {
	proc := &mlxlmProcess{done: make(chan struct{})}
	close(proc.done)

	AssertNoError(t, proc.Kill())
	AssertNoError(t, proc.Wait())
}

func TestAX7_Process_Wait_Good(t *T) {
	proc := &mlxlmProcess{done: make(chan struct{})}
	close(proc.done)

	AssertNoError(t, proc.Wait())
	AssertNil(t, proc.err)
}

func TestAX7_Process_Wait_Bad(t *T) {
	proc := &mlxlmProcess{done: make(chan struct{}), err: NewError("failed")}
	close(proc.done)

	AssertError(t, proc.Wait())
	AssertError(t, proc.err)
}

func TestAX7_Process_Wait_Ugly(t *T) {
	proc := &mlxlmProcess{done: make(chan struct{})}
	close(proc.done)

	AssertNoError(t, proc.Wait())
	AssertNil(t, proc.state)
}

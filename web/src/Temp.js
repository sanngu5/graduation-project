
class App extends Component{

    state = {
      selectedFile: null,
      fileUploadedSuccessfully: false,
      fileChangedSuccessfully: false
    }
    onFileChange = event => {
      if (event.target.files[0].type === 'video/mp4') {
        this.setState({selectedFile : event.target.files[0]});
      } else {
        alert('mp4파일만 변환 가능합니다.')
        this.setState({selectedFile : null});
        this.setState({fileUploadedSuccessfully: false});
        this.setState({fileChangedSuccessfully: false});
      }
    }
  
    onFileUpload = () => {
      //mp4 파일 flask로 전송 코드
      this.setState({selectedFile: null});
      this.setState({fileUploadedSuccessfully: true});
      this.setState({fileChangedSuccessfully: false});
    }
  
    fileChanged = () => {
      //파일변환 완료에 대한 임시 버튼 코드임.
      this.setState({selectedFile: null});
      this.setState({fileUploadedSuccessfully: false});
      this.setState({fileChangedSuccessfully: true});
    }
  
    onFileDownload = () => {
      //변환된 파일 다운로드 코드
      this.setState({selectedFile: null});
      this.setState({fileUploadedSuccessfully: false});
      this.setState({fileChangedSuccessfully: true});
    }
  
    flaskTest = () => {
      //플라스크 테스트
      
    }
  
    fileData = () => {
      if (this.state.selectedFile){
        return(
        <div>
          <br/>
          <h4>업로드 파일 선택 완료</h4>
          <p>파일명: {this.state.selectedFile.name}</p>
          <p>파일유형: {this.state.selectedFile.type}</p>
        </div>
        )
      } else if (this.state.fileUploadedSuccessfully){
        return(
        <div>
          <br/>
          <h4>파일 변환중</h4> 
        </div>
        )
      } else if (this.state.fileChangedSuccessfully){
        return(
        <div>
          <br/>
          <h4>파일 변환 완료</h4> 
        </div>
        )
      } else{
        return(
        <div>
          <br/>
          <h4>업로드 파일을 선택하세요</h4>
        </div>
        )
      }
    }
  
  
  
    render(){
      return (
        <div className='container'>
          <h2>title</h2>
          <div>
            <br></br>
            <input type = "file" onChange = {this.onFileChange} />
            <button disabled={!this.state.selectedFile} onClick={this.onFileUpload}>
              동영상 변환
            </button>
            <button disabled={!this.state.fileUploadedSuccessfully} onClick={this.fileChanged}>
              변환 완료
            </button>
            <button disabled={!this.state.fileChangedSuccessfully} onClick={this.onFileDownload}>
              동영상 다운로드
            </button>
            <button onClick={this.flaskTest}>
              플라스크 테스트
            </button>
          </div>
          {this.fileData()}
          <br/>
        </div>
      )
    }
  }
  
  export default App;
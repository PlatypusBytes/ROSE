import axios from 'axios'

export default function(segmentId) {
  
  if (!segmentId) {
    return 
  }
 
  return axios({
    method: 'get',
    url: `${ process.env.VUE_APP_API_ENDPOINT }/graph_values?segment_id=${ segmentId }`,
  }).then(({ data }) => {
    console.log('data graph values', JSON.stringify(data))
		return typeof data === 'object' ? data : JSON.parse(data)
	})
}